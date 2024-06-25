1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Sélectionner les features et la cible
features = ['Blue Entropy', 'Green Entropy', 'Red Entropy', 'Area-Perimeter Ratio', 'Contour Perimeter',
            'Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3', 'Hu Moment 4', 'Hu Moment 5', 'Hu Moment 6', 'Hu Moment 7','Symmetry Index', 'Aspect Ratio', 'Pixel Ratio']

X = data[features].copy()  # Utiliser .copy() pour éviter SettingWithCopyWarning
y = data['bug type']

# Gestion des valeurs manquantes
X.fillna(X.mean(), inplace=True)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Régression Logistique
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
print("Régression Logistique Classification Report:")
print(classification_report(y_test, log_reg_pred, zero_division=1))

# Support Vector Machine
svm = SVC(kernel='linear')  # Vous pouvez changer le noyau ici
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred, zero_division=1))


2) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialisation du modèle Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # Vous pouvez ajuster le nombre d'arbres

# Entraînement du modèle
random_forest.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
rf_predictions = random_forest.predict(X_test_scaled)

# Évaluation du modèle
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions, zero_division=1))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

3) 
a)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sélection des données pour le clustering
X_cluster = scaler.transform(X)  # Utiliser les données normalisées

# Application de K-Means
kmeans = KMeans(n_clusters=2, random_state=42)  # Supposons 3 clusters pour l'exemple
kmeans.fit(X_cluster)

# Visualisation des clusters
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Première Feature')
plt.ylabel('Deuxième Feature')
plt.title('K-Means Clustering')
plt.colorbar()
plt.show()

b)
from sklearn.cluster import DBSCAN

# Application de DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps et min_samples à ajuster selon vos données
dbscan.fit(X_cluster)

# Visualisation des clusters
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=dbscan.labels_, cmap='viridis')
plt.xlabel('Première Feature')
plt.ylabel('Deuxième Feature')
plt.title('DBSCAN Clustering')
plt.colorbar()
plt.show()




knn)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Initialisation du modèle KNN avec k voisins
knn = KNeighborsClassifier(n_neighbors=12)  # Vous pouvez expérimenter avec différents k

# Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
knn_predictions = knn.predict(X_test_scaled)

# Évaluation du modèle
print("KNN Classification Report:")




hierarchial)
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Define the model
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Plotting the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')
plt.colorbar(label="Cluster ID")
plt.show()
print(classification_report(y_test, knn_predictions, zero_division=1))
