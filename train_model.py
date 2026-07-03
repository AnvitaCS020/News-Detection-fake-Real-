# =============================================================================
#  PROJECT : NEWS ARTICLE CLASSIFICATION — Fake vs Real
#  Dataset : Kaggle "Fake and Real News Dataset"
#            Fake.csv  (23,481 articles) — label = 0
#            True.csv  (21,417 articles) — label = 1
#  Tools   : Python, Pandas, NLTK, Scikit-Learn, Matplotlib, Seaborn
# =============================================================================
#
#  HOW TO RUN
#  ----------
#  Step 1 — Install libraries (run once in terminal):
#      pip install pandas numpy matplotlib seaborn scikit-learn nltk
#
#  Step 2 — Place Fake.csv and True.csv in the SAME folder as this script
#
#  Step 3 — Run:
#      python fake_news_classifier.py
#
#  Output files created in the same folder:
#      saved_model.pkl            ← trained model + vectorizer
#      fig_01_label_dist.png      ← dataset overview
#      fig_02_subject_dist.png    ← subject breakdown
#      fig_03_wordcount.png       ← word length analysis
#      fig_04_confusion_matrix.png
#      fig_05_roc_curve.png
#      fig_06_model_comparison.png
#      fig_07_feature_importance.png
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
import os, re, pickle, warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — works in terminal without a display
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

from sklearn.model_selection     import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.naive_bayes         import MultinomialNB
from sklearn.metrics             import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)

print("=" * 65)
print("   NEWS ARTICLE CLASSIFICATION  —  Fake  vs  Real")
print("=" * 65)
print("\n[STEP 1] All libraries imported.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DOWNLOAD NLTK DATA (runs once, then cached)
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 2] Downloading NLTK data ...")
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)
print("         Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 3] Loading Fake.csv and True.csv ...")

# ── locate files ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

fake_path = os.path.join(SCRIPT_DIR, 'data/Fake.csv')
true_path = os.path.join(SCRIPT_DIR, 'data/True.csv')

if not os.path.exists(fake_path):
    raise FileNotFoundError(f"Fake.csv not found at: {fake_path}\n"
                            "Place Fake.csv and True.csv in the same folder as this script.")
if not os.path.exists(true_path):
    raise FileNotFoundError(f"True.csv not found at: {true_path}")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print(f"         Fake.csv → {fake_df.shape[0]:,} rows, columns: {fake_df.columns.tolist()}")
print(f"         True.csv → {true_df.shape[0]:,} rows, columns: {true_df.columns.tolist()}")

# ── assign labels ─────────────────────────────────────────────────────────────
# 0 = Fake,  1 = Real
fake_df['label'] = 0
true_df['label'] = 1

# ── combine title + text into one column ─────────────────────────────────────
# Both CSVs have 'title' and 'text'. Combining gives more signal.
fake_df['content'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
true_df['content'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')

# ── merge & shuffle ───────────────────────────────────────────────────────────
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[['title', 'text', 'subject', 'date', 'content', 'label']]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── drop empty rows ───────────────────────────────────────────────────────────
df.dropna(subset=['content'], inplace=True)
df = df[df['content'].str.strip() != ''].reset_index(drop=True)

print(f"\n         Combined dataset : {len(df):,} articles")
print(f"         Fake articles   : {(df['label']==0).sum():,}")
print(f"         Real articles   : {(df['label']==1).sum():,}")
print(f"\n         Sample fake title: {df[df['label']==0]['title'].iloc[0][:80]}")
print(f"         Sample real title: {df[df['label']==1]['title'].iloc[0][:80]}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — EXPLORATORY DATA ANALYSIS  (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 4] Running EDA and saving charts ...")

df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['content'].apply(lambda x: len(str(x)))

# ── Figure 1 : Label Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Dataset Overview', fontsize=15, fontweight='bold')

label_counts = df['label'].map({0: 'Fake', 1: 'Real'}).value_counts()
axes[0].pie(
    label_counts,
    labels=label_counts.index,
    autopct='%1.1f%%',
    colors=['#e74c3c', '#27ae60'],
    startangle=90,
    textprops={'fontsize': 13, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('Label Distribution\n(Fake vs Real)', fontsize=12, fontweight='bold')

axes[1].bar(
    ['Fake News', 'Real News'],
    [df[df['label']==0]['word_count'].mean(),
     df[df['label']==1]['word_count'].mean()],
    color=['#e74c3c', '#27ae60'],
    edgecolor='white', linewidth=1.5, width=0.5
)
axes[1].set_ylabel('Average Word Count', fontsize=11)
axes[1].set_title('Average Article Length\n(words per article)', fontsize=12, fontweight='bold')
for i, v in enumerate([df[df['label']==0]['word_count'].mean(),
                        df[df['label']==1]['word_count'].mean()]):
    axes[1].text(i, v + 5, f'{v:.0f}', ha='center', fontsize=12, fontweight='bold')
axes[1].set_facecolor('#f5f5f5')
axes[1].grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_01_label_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_01_label_dist.png")

# ── Figure 2 : Subject Distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Subject Distribution per Label', fontsize=14, fontweight='bold')

for ax, label_val, title, color in [
        (axes[0], 0, 'Fake News — Subject Distribution', '#e74c3c'),
        (axes[1], 1, 'Real News — Subject Distribution', '#27ae60')]:
    counts = df[df['label']==label_val]['subject'].value_counts().head(8)
    ax.barh(counts.index, counts.values, color=color, edgecolor='white',
            linewidth=1.2, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of Articles', fontsize=10)
    ax.set_facecolor('#f5f5f5')
    ax.grid(True, axis='x', alpha=0.4)
    for i, v in enumerate(counts.values):
        ax.text(v + 20, i, f'{v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_02_subject_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_02_subject_dist.png")

# ── Figure 3 : Word Count Histogram ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for label_val, color, name in [(0, '#e74c3c', 'Fake'), (1, '#27ae60', 'Real')]:
    data = df[df['label']==label_val]['word_count'].clip(upper=1500)
    ax.hist(data, bins=50, color=color, alpha=0.60, label=name, edgecolor='white')
ax.set_xlabel('Word Count (clipped at 1500)', fontsize=11)
ax.set_ylabel('Number of Articles', fontsize=11)
ax.set_title('Word Count Distribution — Fake vs Real News', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_facecolor('#f5f5f5')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_03_wordcount.png'), dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_03_wordcount.png\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 5] Preprocessing text (this takes ~1-2 minutes for 44k articles) ...")

STOP_WORDS  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def clean_text(text):
    """
    Pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove non-letter characters (keep only a-z)
    4. Tokenise
    5. Remove stopwords
    6. Lemmatize each token
    7. Drop tokens shorter than 3 characters
    """
    text = str(text).lower()                              # 1. lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)    # 2. remove URLs
    text = re.sub(r'\[.*?\]', '', text)                   #    remove [text]
    text = re.sub(r'[^a-z\s]', '', text)                  # 3. letters only
    tokens = text.split()                                  # 4. tokenise
    tokens = [
        lemmatizer.lemmatize(tok)                          # 6. lemmatize
        for tok in tokens
        if tok not in STOP_WORDS                           # 5. stopwords
        and len(tok) > 2                                   # 7. min length
    ]
    return ' '.join(tokens)

df['cleaned'] = df['content'].apply(clean_text)

# Drop rows where cleaning left empty string
df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)

print(f"         Original  : {df['content'].iloc[0][:80]}...")
print(f"         Cleaned   : {df['cleaned'].iloc[0][:80]}...")
print(f"         Total usable rows after cleaning: {len(df):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 6] Splitting into train (80%) and test (20%) sets ...")

X = df['cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.20,
    random_state= 42,
    stratify    = y      # keeps class ratio equal in both splits
)

print(f"         Training samples : {len(X_train):,}")
print(f"         Testing  samples : {len(X_test):,}")
print(f"         Train — Fake: {sum(y_train==0):,}  Real: {sum(y_train==1):,}")
print(f"         Test  — Fake: {sum(y_test==0):,}   Real: {sum(y_test==1):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — TF-IDF VECTORIZATION
# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF = Term Frequency × Inverse Document Frequency
#   • Words that appear often in ONE article → high TF
#   • Words that appear in MANY articles     → low IDF (penalised)
# ngram_range=(1,2) captures single words AND two-word phrases
# max_features=50000 keeps the 50k most informative terms

print("[STEP 7] Applying TF-IDF Vectorization ...")

tfidf = TfidfVectorizer(
    max_features = 50000,
    ngram_range  = (1, 2),   # unigrams + bigrams
    sublinear_tf = True,     # apply log(1 + tf) — reduces skew
    min_df       = 2,        # ignore terms that appear in < 2 docs
    max_df       = 0.95,     # ignore terms in > 95% of docs (too common)
    analyzer     = 'word',
    strip_accents= 'unicode',
)

# IMPORTANT: fit ONLY on training data to prevent data leakage
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"         Train matrix : {X_train_tfidf.shape}  "
      f"({X_train_tfidf.nnz:,} non-zero values)")
print(f"         Test  matrix : {X_test_tfidf.shape}")
print(f"         Vocabulary   : {len(tfidf.vocabulary_):,} terms\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — TRAIN MODEL A : LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 8] Training Logistic Regression ...")

lr = LogisticRegression(
    C           = 5.0,       # inverse regularization strength
    max_iter    = 1000,
    solver      = 'lbfgs',
    random_state= 42,
    n_jobs      = -1
)
lr.fit(X_train_tfidf, y_train)

lr_train_pred = lr.predict(X_train_tfidf)
lr_test_pred  = lr.predict(X_test_tfidf)

lr_train_acc  = accuracy_score(y_train, lr_train_pred)
lr_test_acc   = accuracy_score(y_test,  lr_test_pred)

print(f"         Train accuracy : {lr_train_acc:.4f}  ({lr_train_acc*100:.2f}%)")
print(f"         Test  accuracy : {lr_test_acc:.4f}  ({lr_test_acc*100:.2f}%)")

# 5-fold cross-validation on training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv = cross_val_score(lr, X_train_tfidf, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"         5-Fold CV      : {[f'{s:.4f}' for s in lr_cv]}")
print(f"         CV Mean ± Std  : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 — TRAIN MODEL B : NAIVE BAYES
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 9] Training Naive Bayes ...")

nb = MultinomialNB(alpha=0.1)    # alpha = Laplace smoothing factor
nb.fit(X_train_tfidf, y_train)

nb_train_pred = nb.predict(X_train_tfidf)
nb_test_pred  = nb.predict(X_test_tfidf)

nb_train_acc  = accuracy_score(y_train, nb_train_pred)
nb_test_acc   = accuracy_score(y_test,  nb_test_pred)

print(f"         Train accuracy : {nb_train_acc:.4f}  ({nb_train_acc*100:.2f}%)")
print(f"         Test  accuracy : {nb_test_acc:.4f}  ({nb_test_acc*100:.2f}%)")

nb_cv = cross_val_score(nb, X_train_tfidf, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"         5-Fold CV      : {[f'{s:.4f}' for s in nb_cv]}")
print(f"         CV Mean ± Std  : {nb_cv.mean():.4f} ± {nb_cv.std():.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 — EVALUATION METRICS (printed to terminal)
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 10] Full Evaluation Metrics\n")

for name, pred in [('Logistic Regression', lr_test_pred),
                   ('Naive Bayes',         nb_test_pred)]:
    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='weighted')
    rec  = recall_score(y_test, pred, average='weighted')
    f1   = f1_score(y_test, pred, average='weighted')
    print(f"  ── {name} ──────────────────────────────")
    print(f"     Accuracy  : {acc:.4f}   ({acc*100:.2f}%)")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print()
    print("     Detailed Report:")
    print(classification_report(y_test, pred, target_names=['Fake', 'Real']))
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 — CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 11] Plotting Confusion Matrices ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrix — Logistic Regression vs Naive Bayes',
             fontsize=14, fontweight='bold')
fig.patch.set_facecolor('#f8f9fa')

for ax, pred, name, cmap in zip(
        axes,
        [lr_test_pred, nb_test_pred],
        ['Logistic Regression', 'Naive Bayes'],
        ['Blues', 'Greens']):

    cm  = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)

    sns.heatmap(
        cm, annot=True, fmt=',d', cmap=cmap, ax=ax,
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real'],
        linewidths=2, linecolor='white',
        annot_kws={'size': 18, 'weight': 'bold'}
    )
    ax.set_title(f'{name}\nAccuracy: {acc:.4f}  ({acc*100:.2f}%)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)

    # TN / FP / FN / TP corner labels
    for i, row in enumerate(['Actual: Fake', 'Actual: Real']):
        for j, col in enumerate(['Pred: Fake', 'Pred: Real']):
            tag = {(0,0):'TN', (0,1):'FP', (1,0):'FN', (1,1):'TP'}[(i,j)]
            ax.text(j+0.5, i+0.78, f'({tag})',
                    ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_04_confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_04_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 12 — ROC CURVE PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 12] Plotting ROC Curves ...")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f0f4f8')

for model, name, color in [(lr, 'Logistic Regression', '#2980b9'),
                            (nb, 'Naive Bayes',         '#e74c3c')]:
    proba   = model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f'{name}  (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.06, color=color)

ax.plot([0,1], [0,1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5000)')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.03])
ax.set_xlabel('False Positive Rate  (FPR)', fontsize=11)
ax.set_ylabel('True Positive Rate  (TPR = Recall)', fontsize=11)
ax.set_title('ROC Curve — Fake News Detection', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_05_roc_curve.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_05_roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 13 — MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 13] Plotting Model Comparison Chart ...")

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_scores = [
    accuracy_score(y_test,  lr_test_pred),
    precision_score(y_test, lr_test_pred, average='weighted'),
    recall_score(y_test,    lr_test_pred, average='weighted'),
    f1_score(y_test,        lr_test_pred, average='weighted'),
]
nb_scores = [
    accuracy_score(y_test,  nb_test_pred),
    precision_score(y_test, nb_test_pred, average='weighted'),
    recall_score(y_test,    nb_test_pred, average='weighted'),
    f1_score(y_test,        nb_test_pred, average='weighted'),
]

x = np.arange(len(metric_names))
w = 0.30

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f0f4f8')

b1 = ax.bar(x - w/2, lr_scores, w, label='Logistic Regression',
            color='#2980b9', edgecolor='white', linewidth=1.5, alpha=0.88)
b2 = ax.bar(x + w/2, nb_scores, w, label='Naive Bayes',
            color='#e74c3c', edgecolor='white', linewidth=1.5, alpha=0.88)

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_06_model_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_06_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 14 — FEATURE IMPORTANCE PLOT (top TF-IDF words per class)
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 14] Plotting Feature Importance ...")

feature_names = np.array(tfidf.get_feature_names_out())
coef          = lr.coef_[0]          # one coefficient per TF-IDF feature
#  negative coef → predicts Fake (class 0)
#  positive coef → predicts Real (class 1)

TOP_N = 20
top_fake_idx  = np.argsort(coef)[:TOP_N]
top_real_idx  = np.argsort(coef)[-TOP_N:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 20 Most Discriminating TF-IDF Features\n(Logistic Regression Coefficients)',
             fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#f8f9fa')

for ax, idx, title, color in [
        (axes[0], top_fake_idx, 'Top 20 → FAKE News', '#e74c3c'),
        (axes[1], top_real_idx, 'Top 20 → REAL News', '#27ae60')]:
    ax.set_facecolor('#f0f4f8')
    vals  = np.abs(coef[idx])
    words = feature_names[idx]
    bars  = ax.barh(range(len(words)), vals, color=color, alpha=0.82,
                    edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel('|Coefficient| (feature importance)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color)
    ax.grid(True, axis='x', alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8, color='#555555')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'fig_07_feature_importance.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: fig_07_feature_importance.png\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 15 — SAVE MODEL TO DISK
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 15] Saving model to disk ...")

best_model = lr if lr_test_acc >= nb_test_acc else nb
best_name  = 'Logistic Regression' if lr_test_acc >= nb_test_acc else 'Naive Bayes'

payload = {
    'model':      best_model,     # trained classifier
    'vectorizer': tfidf,          # MUST save — defines the feature space
    'model_name': best_name,
    'accuracy':   max(lr_test_acc, nb_test_acc),
    'label_map':  {0: 'FAKE', 1: 'REAL'},
}

model_path = os.path.join(SCRIPT_DIR, 'saved_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(payload, f)

print(f"         Best model : {best_name}")
print(f"         Accuracy   : {payload['accuracy']:.4f}  ({payload['accuracy']*100:.2f}%)")
print(f"         Saved to   : {model_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 16 — LIVE PREDICTION DEMO (uses loaded model from disk)
# ─────────────────────────────────────────────────────────────────────────────
print("[STEP 16] Running live prediction demo ...\n")

# --- load from disk (simulates what the Streamlit app will do) ---
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)

def predict(raw_text):
    """
    Full pipeline:  raw text → preprocess → TF-IDF → predict
    Returns (label_str, confidence_pct, fake_pct, real_pct)
    """
    cleaned = clean_text(raw_text)
    vec     = loaded['vectorizer'].transform([cleaned])
    pred    = loaded['model'].predict(vec)[0]
    proba   = loaded['model'].predict_proba(vec)[0]
    label   = loaded['label_map'][pred]
    return label, max(proba)*100, proba[0]*100, proba[1]*100

test_articles = [
    ("SHOCKING: Government secretly adding mind-control chemicals to drinking water. "
     "Whistleblowers have come forward with explosive evidence. Share before deleted!",
     "Expected → FAKE"),

    ("WASHINGTON (Reuters) - The Federal Reserve raised its benchmark interest rate "
     "by a quarter of a percentage point on Wednesday, as widely expected by analysts "
     "and market participants.",
     "Expected → REAL"),

    ("NASA admits Moon landing was faked in Hollywood studio. Anonymous source reveals "
     "the truth the deep state has hidden for decades. The mainstream media won't cover this.",
     "Expected → FAKE"),

    ("The Senate passed a bipartisan infrastructure bill on Tuesday after months of "
     "negotiations. The legislation includes funding for roads, bridges and broadband.",
     "Expected → REAL"),
]

print("  " + "─" * 62)
for article, expected in test_articles:
    label, conf, fake_pct, real_pct = predict(article)
    icon = "✓" if label == "REAL" else "✗"
    print(f"\n  Article  : {article[:75]}...")
    print(f"  {expected}")
    print(f"  Result   : [{icon}] {label}   Confidence: {conf:.1f}%")
    print(f"             Fake prob: {fake_pct:.1f}%   |   Real prob: {real_pct:.1f}%")
    print("  " + "─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 17 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   FINAL SUMMARY")
print("=" * 65)
rows = [
    ("Total articles",          f"{len(df):,}  (Fake: {(df['label']==0).sum():,}  Real: {(df['label']==1).sum():,})"),
    ("TF-IDF vocabulary",       f"{len(tfidf.vocabulary_):,} features (unigrams + bigrams)"),
    ("Train / Test split",      "80% / 20%  (stratified)"),
    ("LR  — Test Accuracy",     f"{lr_test_acc:.4f}  ({lr_test_acc*100:.2f}%)"),
    ("LR  — Test F1-Score",     f"{f1_score(y_test, lr_test_pred, average='weighted'):.4f}"),
    ("LR  — 5-Fold CV Mean",    f"{lr_cv.mean():.4f} ± {lr_cv.std():.4f}"),
    ("NB  — Test Accuracy",     f"{nb_test_acc:.4f}  ({nb_test_acc*100:.2f}%)"),
    ("NB  — Test F1-Score",     f"{f1_score(y_test, nb_test_pred, average='weighted'):.4f}"),
    ("NB  — 5-Fold CV Mean",    f"{nb_cv.mean():.4f} ± {nb_cv.std():.4f}"),
    ("Best Model",              best_name),
    ("Model saved to",          "saved_model.pkl"),
]
for k, v in rows:
    print(f"   {k:<30} : {v}")
print()
print("   Charts saved:")
for i, name in enumerate([
    'fig_01_label_dist.png', 'fig_02_subject_dist.png', 'fig_03_wordcount.png',
    'fig_04_confusion_matrix.png', 'fig_05_roc_curve.png',
    'fig_06_model_comparison.png', 'fig_07_feature_importance.png'], 1):
    print(f"     {i}. {name}")
print()
print("   Next step → run Streamlit app:  streamlit run app.py")
print("=" * 65)