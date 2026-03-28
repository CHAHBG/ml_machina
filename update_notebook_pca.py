import json

with open("EDA_Maintenance_Predictive.ipynb", "r") as f:
    nb = json.load(f)

# Find the markdown cell for ACP
acp_md_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "Analyse en Composantes Principales" in src and "ACP" in src:
            acp_md_idx = i
            break

if acp_md_idx != -1:
    new_md = """## 10. Analyse en Composantes Principales (A.C.P.)

Conformement a la theorie mathematique, l'**A.C.P.** permet d'analyser des donnees multidimensionnelles 
composees de plusieurs **variables quantitatives correlees**, et de les reduire en un nombre reduit de 
**nouvelles variables decorrelees** appelees "composantes principales", tout en conservant le maximum 
d'information (variance).

### Processus theorique applique :
1. **Donnees centrees-reduites** : Pour annuler l'effet des unites, toutes les variables sont standardisees (moyenne = 0, ecart-type = 1).
2. **Inertie totale** : Elle mesure la dispersion globale du nuage de points et correspond a la trace de la matrice de variances-covariances.
3. **Axes principaux** : Ce sont les vecteurs propres de la matrice de variances-covariances, ordonnes par valeur propre (variance) decroissante.
4. **Composantes principales** : Nouvelles variables (combinaisons lineaires) fournissant les coordonnees sur les axes principaux. Elles sont non-correlees.
5. **Critere global (Regle des 95%)** : On retient les premieres composantes dont la somme cumulee des variances represente au moins **95% de l'inertie totale**.

L'ACP va donc prendre l'ensemble de nos variables (transformees) au lieu d'une selection manuelle, 
afin de trouver les meilleures combinaisons lineaires pour expliquer la panne.
"""
    nb["cells"][acp_md_idx]["source"] = [line + "\n" for line in new_md.split("\n")[:-1]] + [new_md.split("\n")[-1]]

# Find the code cell for ACP
acp_code_idx = -1
for i in range(acp_md_idx+1, len(nb["cells"])):
    if nb["cells"][i]["cell_type"] == "code":
        acp_code_idx = i
        break

if acp_code_idx != -1:
    new_code = """from sklearn.decomposition import PCA

# 1. Preparation des donnees : on utilise TOUTES les colonnes (sauf id, cible, date)
X_pca = df.drop(columns=['id', 'panne', 'date_derniere_maintenance'])

# Identification des colonnes numeriques et categorielles
num_cols_pca = X_pca.select_dtypes(include=['int64', 'float64']).columns
cat_cols_pca = X_pca.select_dtypes(include=['object']).columns

# 2. Pipeline de pretraitement (Imputation + Standardisation/Encodage)
# Les donnees numeriques sont centrees-reduites (StandardScaler)
preprocessor_pca = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols_pca),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols_pca)
    ])

# 3. Application du pretraitement
X_pca_processed = preprocessor_pca.fit_transform(X_pca)

# Noms des colonnes apres encodage One-Hot
feature_names_pca = (num_cols_pca.tolist() + 
                     preprocessor_pca.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols_pca).tolist())

print(f"Dimensions avant ACP : {X_pca.shape[1]} variables -> {X_pca_processed.shape[1]} colonnes (apres encodage)")

# 4. Ajustement de l'ACP sur l'ensemble des donnees
pca = PCA(random_state=42)
pca.fit(X_pca_processed)

# 5. Calcul de la variance cumulee et regle des 95%
cumul_var = np.cumsum(pca.explained_variance_ratio_) * 100
n_components_95 = np.argmax(cumul_var >= 95) + 1

print(f"Nombre de composantes pour capturer 95% de l'inertie totale : {n_components_95}")

# Affichage du Scree Plot (Variance individuelle et cumulee)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100, alpha=0.7, color='blue')
plt.ylabel('Variance expliquee (%)')
plt.xlabel('Composante principale')
plt.title('Inertie par axe principal (Top 20)')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumul_var) + 1), cumul_var, marker='o', markersize=4, linestyle='-', color='red')
plt.axhline(y=95, color='k', linestyle='--', label='Seuil 95%')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} composantes')
plt.ylabel('Variance cumulee (%)')
plt.xlabel('Nombre de composantes')
plt.title('Inertie totale cumulee')
plt.legend()
plt.tight_layout()
plt.show()

# Affichage des poids (vecteurs propres) des variables sur la premiere composante
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
    index=feature_names_pca
)
print("\\nVariables contribuant le plus a l'axe 1 (PC1) :")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))
"""
    nb["cells"][acp_code_idx]["source"] = [line + "\n" for line in new_code.split("\n")[:-1]] + [new_code.split("\n")[-1]]


# The SVM section (Section 11) is next. We should also update it to use X_pca_processed.
svm_md_idx = -1
for i in range(acp_code_idx+1, len(nb["cells"])):
    if nb["cells"][i]["cell_type"] == "markdown":
        src = "".join(nb["cells"][i]["source"])
        if "Modélisation SVM" in src or "Modelisation SVM" in src:
            svm_md_idx = i
            break

if svm_md_idx != -1:
    svm_code_idx = -1
    for i in range(svm_md_idx+1, len(nb["cells"])):
        if nb["cells"][i]["cell_type"] == "code":
            svm_code_idx = i
            break
            
    if svm_code_idx != -1:
        new_svm_code = """from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# 1. Separation entrainement / test
y = df['panne']
X_train, X_test, y_train, y_test = train_test_split(X_pca_processed, y, test_size=0.2, random_state=42, stratify=y)

# 2. Application de l'ACP limitee a 95% de variance
pca_final = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca_final.fit_transform(X_train)
X_test_pca = pca_final.transform(X_test)

# 3. Entrainement du modele SVM (Support Vector Machine)
# Kernel RBF pour capturer les relations non lineaires
# class_weight='balanced' pour gerer le desequilibre (16% de pannes)
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train_pca, y_train)

# 4. Predictions et Evaluation
y_pred_svm = svm_model.predict(X_test_pca)

print("=== Rapport de Classification (SVM sur Composantes Principales) ===")
print( वर्गीकरण_report := classification_report(y_test, y_pred_svm) )

# Matrice de confusion
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sain Predit', 'Panne Predite'], 
            yticklabels=['Sain Reel', 'Panne Reelle'])
plt.title('Matrice de Confusion (SVM)')
plt.show()
"""
        nb["cells"][svm_code_idx]["source"] = [line + "\n" for line in new_svm_code.split("\n")[:-1]] + [new_svm_code.split("\n")[-1]]


with open("EDA_Maintenance_Predictive.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
