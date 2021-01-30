{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3",
   "language": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])"
      ]
     },
     "metadata": {},
     "execution_count": 1
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "\n",
    "\n",
    "cancer = load_breast_cancer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])"
      ]
     },
     "metadata": {},
     "execution_count": 2
    }
   ],
   "source": [
    "cancer.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      ".. _breast_cancer_dataset:\n\nBreast cancer wisconsin (diagnostic) dataset\n--------------------------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 569\n\n    :Number of Attributes: 30 numeric, predictive attributes and the class\n\n    :Attribute Information:\n        - radius (mean of distances from center to points on the perimeter)\n        - texture (standard deviation of gray-scale values)\n        - perimeter\n        - area\n        - smoothness (local variation in radius lengths)\n        - compactness (perimeter^2 / area - 1.0)\n        - concavity (severity of concave portions of the contour)\n        - concave points (number of concave portions of the contour)\n        - symmetry\n        - fractal dimension (\"coastline approximation\" - 1)\n\n        The mean, standard error, and \"worst\" or largest (mean of the three\n        worst/largest values) of these features were computed for each image,\n        resulting in 30 features.  For instance, field 0 is Mean Radius, field\n        10 is Radius SE, field 20 is Worst Radius.\n\n        - class:\n                - WDBC-Malignant\n                - WDBC-Benign\n\n    :Summary Statistics:\n\n    ===================================== ====== ======\n                                           Min    Max\n    ===================================== ====== ======\n    radius (mean):                        6.981  28.11\n    texture (mean):                       9.71   39.28\n    perimeter (mean):                     43.79  188.5\n    area (mean):                          143.5  2501.0\n    smoothness (mean):                    0.053  0.163\n    compactness (mean):                   0.019  0.345\n    concavity (mean):                     0.0    0.427\n    concave points (mean):                0.0    0.201\n    symmetry (mean):                      0.106  0.304\n    fractal dimension (mean):             0.05   0.097\n    radius (standard error):              0.112  2.873\n    texture (standard error):             0.36   4.885\n    perimeter (standard error):           0.757  21.98\n    area (standard error):                6.802  542.2\n    smoothness (standard error):          0.002  0.031\n    compactness (standard error):         0.002  0.135\n    concavity (standard error):           0.0    0.396\n    concave points (standard error):      0.0    0.053\n    symmetry (standard error):            0.008  0.079\n    fractal dimension (standard error):   0.001  0.03\n    radius (worst):                       7.93   36.04\n    texture (worst):                      12.02  49.54\n    perimeter (worst):                    50.41  251.2\n    area (worst):                         185.2  4254.0\n    smoothness (worst):                   0.071  0.223\n    compactness (worst):                  0.027  1.058\n    concavity (worst):                    0.0    1.252\n    concave points (worst):               0.0    0.291\n    symmetry (worst):                     0.156  0.664\n    fractal dimension (worst):            0.055  0.208\n    ===================================== ====== ======\n\n    :Missing Attribute Values: None\n\n    :Class Distribution: 212 - Malignant, 357 - Benign\n\n    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\n\n    :Donor: Nick Street\n\n    :Date: November, 1995\n\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\nhttps://goo.gl/U2Uwz2\n\nFeatures are computed from a digitized image of a fine needle\naspirate (FNA) of a breast mass.  They describe\ncharacteristics of the cell nuclei present in the image.\n\nSeparating plane described above was obtained using\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, \"Decision Tree\nConstruction Via Linear Programming.\" Proceedings of the 4th\nMidwest Artificial Intelligence and Cognitive Science Society,\npp. 97-101, 1992], a classification method which uses linear\nprogramming to construct a decision tree.  Relevant features\nwere selected using an exhaustive search in the space of 1-4\nfeatures and 1-3 separating planes.\n\nThe actual linear program used to obtain the separating plane\nin the 3-dimensional space is that described in:\n[K. P. Bennett and O. L. Mangasarian: \"Robust Linear\nProgramming Discrimination of Two Linearly Inseparable Sets\",\nOptimization Methods and Software 1, 1992, 23-34].\n\nThis database is also available through the UW CS ftp server:\n\nftp ftp.cs.wisc.edu\ncd math-prog/cpo-dataset/machine-learn/WDBC/\n\n.. topic:: References\n\n   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \n     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \n     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\n     San Jose, CA, 1993.\n   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \n     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \n     July-August 1995.\n   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\n     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \n     163-171.\n"
     ]
    }
   ],
   "source": [
    "print(cancer['DESCR'])\n",
    "# We have 30 numerical, predictive attributes\n",
    "# 569 instances aka rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "sklearn.utils.Bunch"
      ]
     },
     "metadata": {},
     "execution_count": 6
    }
   ],
   "source": [
    "type(cancer)\n",
    "# Bunch type data is a container object which shows attributes as keys \n",
    "# We will need to convert this data into a DF to be able to apply pandas on it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option('display.max_columns', 500)\n",
    "pd.set_option('display.max_rows', 500)\n",
    "\n",
    "# We set number of max rows and max columns to 500 so that we can see all the columns without Python collapsing them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
       "0        17.99         10.38          122.80     1001.0          0.11840   \n",
       "1        20.57         17.77          132.90     1326.0          0.08474   \n",
       "2        19.69         21.25          130.00     1203.0          0.10960   \n",
       "3        11.42         20.38           77.58      386.1          0.14250   \n",
       "4        20.29         14.34          135.10     1297.0          0.10030   \n",
       "\n",
       "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
       "0           0.27760          0.3001              0.14710         0.2419   \n",
       "1           0.07864          0.0869              0.07017         0.1812   \n",
       "2           0.15990          0.1974              0.12790         0.2069   \n",
       "3           0.28390          0.2414              0.10520         0.2597   \n",
       "4           0.13280          0.1980              0.10430         0.1809   \n",
       "\n",
       "   mean fractal dimension  radius error  texture error  perimeter error  \\\n",
       "0                 0.07871        1.0950         0.9053            8.589   \n",
       "1                 0.05667        0.5435         0.7339            3.398   \n",
       "2                 0.05999        0.7456         0.7869            4.585   \n",
       "3                 0.09744        0.4956         1.1560            3.445   \n",
       "4                 0.05883        0.7572         0.7813            5.438   \n",
       "\n",
       "   area error  smoothness error  compactness error  concavity error  \\\n",
       "0      153.40          0.006399            0.04904          0.05373   \n",
       "1       74.08          0.005225            0.01308          0.01860   \n",
       "2       94.03          0.006150            0.04006          0.03832   \n",
       "3       27.23          0.009110            0.07458          0.05661   \n",
       "4       94.44          0.011490            0.02461          0.05688   \n",
       "\n",
       "   concave points error  symmetry error  fractal dimension error  \\\n",
       "0               0.01587         0.03003                 0.006193   \n",
       "1               0.01340         0.01389                 0.003532   \n",
       "2               0.02058         0.02250                 0.004571   \n",
       "3               0.01867         0.05963                 0.009208   \n",
       "4               0.01885         0.01756                 0.005115   \n",
       "\n",
       "   worst radius  worst texture  worst perimeter  worst area  worst smoothness  \\\n",
       "0         25.38          17.33           184.60      2019.0            0.1622   \n",
       "1         24.99          23.41           158.80      1956.0            0.1238   \n",
       "2         23.57          25.53           152.50      1709.0            0.1444   \n",
       "3         14.91          26.50            98.87       567.7            0.2098   \n",
       "4         22.54          16.67           152.20      1575.0            0.1374   \n",
       "\n",
       "   worst compactness  worst concavity  worst concave points  worst symmetry  \\\n",
       "0             0.6656           0.7119                0.2654          0.4601   \n",
       "1             0.1866           0.2416                0.1860          0.2750   \n",
       "2             0.4245           0.4504                0.2430          0.3613   \n",
       "3             0.8663           0.6869                0.2575          0.6638   \n",
       "4             0.2050           0.4000                0.1625          0.2364   \n",
       "\n",
       "   worst fractal dimension  \n",
       "0                  0.11890  \n",
       "1                  0.08902  \n",
       "2                  0.08758  \n",
       "3                  0.17300  \n",
       "4                  0.07678  "
      ],
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>mean radius</th>\n      <th>mean texture</th>\n      <th>mean perimeter</th>\n      <th>mean area</th>\n      <th>mean smoothness</th>\n      <th>mean compactness</th>\n      <th>mean concavity</th>\n      <th>mean concave points</th>\n      <th>mean symmetry</th>\n      <th>mean fractal dimension</th>\n      <th>radius error</th>\n      <th>texture error</th>\n      <th>perimeter error</th>\n      <th>area error</th>\n      <th>smoothness error</th>\n      <th>compactness error</th>\n      <th>concavity error</th>\n      <th>concave points error</th>\n      <th>symmetry error</th>\n      <th>fractal dimension error</th>\n      <th>worst radius</th>\n      <th>worst texture</th>\n      <th>worst perimeter</th>\n      <th>worst area</th>\n      <th>worst smoothness</th>\n      <th>worst compactness</th>\n      <th>worst concavity</th>\n      <th>worst concave points</th>\n      <th>worst symmetry</th>\n      <th>worst fractal dimension</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>17.99</td>\n      <td>10.38</td>\n      <td>122.80</td>\n      <td>1001.0</td>\n      <td>0.11840</td>\n      <td>0.27760</td>\n      <td>0.3001</td>\n      <td>0.14710</td>\n      <td>0.2419</td>\n      <td>0.07871</td>\n      <td>1.0950</td>\n      <td>0.9053</td>\n      <td>8.589</td>\n      <td>153.40</td>\n      <td>0.006399</td>\n      <td>0.04904</td>\n      <td>0.05373</td>\n      <td>0.01587</td>\n      <td>0.03003</td>\n      <td>0.006193</td>\n      <td>25.38</td>\n      <td>17.33</td>\n      <td>184.60</td>\n      <td>2019.0</td>\n      <td>0.1622</td>\n      <td>0.6656</td>\n      <td>0.7119</td>\n      <td>0.2654</td>\n      <td>0.4601</td>\n      <td>0.11890</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>20.57</td>\n      <td>17.77</td>\n      <td>132.90</td>\n      <td>1326.0</td>\n      <td>0.08474</td>\n      <td>0.07864</td>\n      <td>0.0869</td>\n      <td>0.07017</td>\n      <td>0.1812</td>\n      <td>0.05667</td>\n      <td>0.5435</td>\n      <td>0.7339</td>\n      <td>3.398</td>\n      <td>74.08</td>\n      <td>0.005225</td>\n      <td>0.01308</td>\n      <td>0.01860</td>\n      <td>0.01340</td>\n      <td>0.01389</td>\n      <td>0.003532</td>\n      <td>24.99</td>\n      <td>23.41</td>\n      <td>158.80</td>\n      <td>1956.0</td>\n      <td>0.1238</td>\n      <td>0.1866</td>\n      <td>0.2416</td>\n      <td>0.1860</td>\n      <td>0.2750</td>\n      <td>0.08902</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>19.69</td>\n      <td>21.25</td>\n      <td>130.00</td>\n      <td>1203.0</td>\n      <td>0.10960</td>\n      <td>0.15990</td>\n      <td>0.1974</td>\n      <td>0.12790</td>\n      <td>0.2069</td>\n      <td>0.05999</td>\n      <td>0.7456</td>\n      <td>0.7869</td>\n      <td>4.585</td>\n      <td>94.03</td>\n      <td>0.006150</td>\n      <td>0.04006</td>\n      <td>0.03832</td>\n      <td>0.02058</td>\n      <td>0.02250</td>\n      <td>0.004571</td>\n      <td>23.57</td>\n      <td>25.53</td>\n      <td>152.50</td>\n      <td>1709.0</td>\n      <td>0.1444</td>\n      <td>0.4245</td>\n      <td>0.4504</td>\n      <td>0.2430</td>\n      <td>0.3613</td>\n      <td>0.08758</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>11.42</td>\n      <td>20.38</td>\n      <td>77.58</td>\n      <td>386.1</td>\n      <td>0.14250</td>\n      <td>0.28390</td>\n      <td>0.2414</td>\n      <td>0.10520</td>\n      <td>0.2597</td>\n      <td>0.09744</td>\n      <td>0.4956</td>\n      <td>1.1560</td>\n      <td>3.445</td>\n      <td>27.23</td>\n      <td>0.009110</td>\n      <td>0.07458</td>\n      <td>0.05661</td>\n      <td>0.01867</td>\n      <td>0.05963</td>\n      <td>0.009208</td>\n      <td>14.91</td>\n      <td>26.50</td>\n      <td>98.87</td>\n      <td>567.7</td>\n      <td>0.2098</td>\n      <td>0.8663</td>\n      <td>0.6869</td>\n      <td>0.2575</td>\n      <td>0.6638</td>\n      <td>0.17300</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>20.29</td>\n      <td>14.34</td>\n      <td>135.10</td>\n      <td>1297.0</td>\n      <td>0.10030</td>\n      <td>0.13280</td>\n      <td>0.1980</td>\n      <td>0.10430</td>\n      <td>0.1809</td>\n      <td>0.05883</td>\n      <td>0.7572</td>\n      <td>0.7813</td>\n      <td>5.438</td>\n      <td>94.44</td>\n      <td>0.011490</td>\n      <td>0.02461</td>\n      <td>0.05688</td>\n      <td>0.01885</td>\n      <td>0.01756</td>\n      <td>0.005115</td>\n      <td>22.54</td>\n      <td>16.67</td>\n      <td>152.20</td>\n      <td>1575.0</td>\n      <td>0.1374</td>\n      <td>0.2050</td>\n      <td>0.4000</td>\n      <td>0.1625</td>\n      <td>0.2364</td>\n      <td>0.07678</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "metadata": {},
     "execution_count": 12
    }
   ],
   "source": [
    "df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We will now standardize our data so that we can apply PCA to it in the next stage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "StandardScaler()"
      ]
     },
     "metadata": {},
     "execution_count": 14
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "scaler.fit(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaled_data = scaler.transform(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "array([[ 1.09706398, -2.07333501,  1.26993369, ...,  2.29607613,\n",
       "         2.75062224,  1.93701461],\n",
       "       [ 1.82982061, -0.35363241,  1.68595471, ...,  1.0870843 ,\n",
       "        -0.24388967,  0.28118999],\n",
       "       [ 1.57988811,  0.45618695,  1.56650313, ...,  1.95500035,\n",
       "         1.152255  ,  0.20139121],\n",
       "       ...,\n",
       "       [ 0.70228425,  2.0455738 ,  0.67267578, ...,  0.41406869,\n",
       "        -1.10454895, -0.31840916],\n",
       "       [ 1.83834103,  2.33645719,  1.98252415, ...,  2.28998549,\n",
       "         1.91908301,  2.21963528],\n",
       "       [-1.80840125,  1.22179204, -1.81438851, ..., -1.74506282,\n",
       "        -0.04813821, -0.75120669]])"
      ]
     },
     "metadata": {},
     "execution_count": 17
    }
   ],
   "source": [
    "scaled_data\n",
    "# Confirming that the data has been scaled by Standardizing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "pca = PCA(n_components=5)\n",
    "# Specifying that we want only 5 Principal Components"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "PCA(n_components=5)"
      ]
     },
     "metadata": {},
     "execution_count": 20
    }
   ],
   "source": [
    "pca.fit(scaled_data)\n",
    "# We first call FIT on the scaled data so that PCA can learn from the entire dataset\n",
    "# Here we fit our standardized data with the PCA object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_pca = pca.transform(scaled_data)\n",
    "# We then let the \"learnt\" PCA tranform our scaled data and we store the result in the object x_pca which has the 5 PCs we requested."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(569, 30)"
      ]
     },
     "metadata": {},
     "execution_count": 22
    }
   ],
   "source": [
    "scaled_data.shape\n",
    "# The scaled data shape has 569 rows and 30 features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(569, 5)"
      ]
     },
     "metadata": {},
     "execution_count": 23
    }
   ],
   "source": [
    "x_pca.shape\n",
    "# The x_pca has 569 rows with 5 Principal Components"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "[0.44272026 0.18971182 0.09393163 0.06602135 0.05495768]\n"
     ]
    }
   ],
   "source": [
    "print(pca.explained_variance_ratio_)\n",
    "# Percentage of variance explained by each of the selected components"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.8473427431680519"
      ]
     },
     "metadata": {},
     "execution_count": 29
    }
   ],
   "source": [
    "print(np.sum(pca.explained_variance_ratio_))\n",
    "# notice that approximately 85% of variance is explained by these 5 PCs. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": "<Figure size 432x288 with 1 Axes>",
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"262.19625pt\" version=\"1.1\" viewBox=\"0 0 392.14375 262.19625\" width=\"392.14375pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <defs>\r\n  <style type=\"text/css\">\r\n*{stroke-linecap:butt;stroke-linejoin:round;}\r\n  </style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M -0 262.19625 \r\nL 392.14375 262.19625 \r\nL 392.14375 0 \r\nL -0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 50.14375 224.64 \r\nL 384.94375 224.64 \r\nL 384.94375 7.2 \r\nL 50.14375 7.2 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m8c12faeeb8\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"65.361932\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 0.0 -->\r\n      <defs>\r\n       <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-46\"/>\r\n      </defs>\r\n      <g transform=\"translate(57.410369 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_2\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"103.407386\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 0.5 -->\r\n      <defs>\r\n       <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-53\"/>\r\n      </defs>\r\n      <g transform=\"translate(95.455824 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_3\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"141.452841\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 1.0 -->\r\n      <defs>\r\n       <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n      </defs>\r\n      <g transform=\"translate(133.501278 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"179.498295\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 1.5 -->\r\n      <g transform=\"translate(171.546733 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_5\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"217.54375\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 2.0 -->\r\n      <defs>\r\n       <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n      </defs>\r\n      <g transform=\"translate(209.592188 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_6\">\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"255.589205\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 2.5 -->\r\n      <g transform=\"translate(247.637642 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_7\">\r\n     <g id=\"line2d_7\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"293.634659\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 3.0 -->\r\n      <defs>\r\n       <path d=\"M 40.578125 39.3125 \r\nQ 47.65625 37.796875 51.625 33 \r\nQ 55.609375 28.21875 55.609375 21.1875 \r\nQ 55.609375 10.40625 48.1875 4.484375 \r\nQ 40.765625 -1.421875 27.09375 -1.421875 \r\nQ 22.515625 -1.421875 17.65625 -0.515625 \r\nQ 12.796875 0.390625 7.625 2.203125 \r\nL 7.625 11.71875 \r\nQ 11.71875 9.328125 16.59375 8.109375 \r\nQ 21.484375 6.890625 26.8125 6.890625 \r\nQ 36.078125 6.890625 40.9375 10.546875 \r\nQ 45.796875 14.203125 45.796875 21.1875 \r\nQ 45.796875 27.640625 41.28125 31.265625 \r\nQ 36.765625 34.90625 28.71875 34.90625 \r\nL 20.21875 34.90625 \r\nL 20.21875 43.015625 \r\nL 29.109375 43.015625 \r\nQ 36.375 43.015625 40.234375 45.921875 \r\nQ 44.09375 48.828125 44.09375 54.296875 \r\nQ 44.09375 59.90625 40.109375 62.90625 \r\nQ 36.140625 65.921875 28.71875 65.921875 \r\nQ 24.65625 65.921875 20.015625 65.03125 \r\nQ 15.375 64.15625 9.8125 62.3125 \r\nL 9.8125 71.09375 \r\nQ 15.4375 72.65625 20.34375 73.4375 \r\nQ 25.25 74.21875 29.59375 74.21875 \r\nQ 40.828125 74.21875 47.359375 69.109375 \r\nQ 53.90625 64.015625 53.90625 55.328125 \r\nQ 53.90625 49.265625 50.4375 45.09375 \r\nQ 46.96875 40.921875 40.578125 39.3125 \r\nz\r\n\" id=\"DejaVuSans-51\"/>\r\n      </defs>\r\n      <g transform=\"translate(285.683097 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-51\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_8\">\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"331.680114\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_8\">\r\n      <!-- 3.5 -->\r\n      <g transform=\"translate(323.728551 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-51\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_9\">\r\n     <g id=\"line2d_9\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"369.725568\" xlink:href=\"#m8c12faeeb8\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 4.0 -->\r\n      <defs>\r\n       <path d=\"M 37.796875 64.3125 \r\nL 12.890625 25.390625 \r\nL 37.796875 25.390625 \r\nz\r\nM 35.203125 72.90625 \r\nL 47.609375 72.90625 \r\nL 47.609375 25.390625 \r\nL 58.015625 25.390625 \r\nL 58.015625 17.1875 \r\nL 47.609375 17.1875 \r\nL 47.609375 0 \r\nL 37.796875 0 \r\nL 37.796875 17.1875 \r\nL 4.890625 17.1875 \r\nL 4.890625 26.703125 \r\nz\r\n\" id=\"DejaVuSans-52\"/>\r\n      </defs>\r\n      <g transform=\"translate(361.774006 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-52\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_10\">\r\n     <!-- Number of Principal Components -->\r\n     <defs>\r\n      <path d=\"M 9.8125 72.90625 \r\nL 23.09375 72.90625 \r\nL 55.421875 11.921875 \r\nL 55.421875 72.90625 \r\nL 64.984375 72.90625 \r\nL 64.984375 0 \r\nL 51.703125 0 \r\nL 19.390625 60.984375 \r\nL 19.390625 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-78\"/>\r\n      <path d=\"M 8.5 21.578125 \r\nL 8.5 54.6875 \r\nL 17.484375 54.6875 \r\nL 17.484375 21.921875 \r\nQ 17.484375 14.15625 20.5 10.265625 \r\nQ 23.53125 6.390625 29.59375 6.390625 \r\nQ 36.859375 6.390625 41.078125 11.03125 \r\nQ 45.3125 15.671875 45.3125 23.6875 \r\nL 45.3125 54.6875 \r\nL 54.296875 54.6875 \r\nL 54.296875 0 \r\nL 45.3125 0 \r\nL 45.3125 8.40625 \r\nQ 42.046875 3.421875 37.71875 1 \r\nQ 33.40625 -1.421875 27.6875 -1.421875 \r\nQ 18.265625 -1.421875 13.375 4.4375 \r\nQ 8.5 10.296875 8.5 21.578125 \r\nz\r\nM 31.109375 56 \r\nz\r\n\" id=\"DejaVuSans-117\"/>\r\n      <path d=\"M 52 44.1875 \r\nQ 55.375 50.25 60.0625 53.125 \r\nQ 64.75 56 71.09375 56 \r\nQ 79.640625 56 84.28125 50.015625 \r\nQ 88.921875 44.046875 88.921875 33.015625 \r\nL 88.921875 0 \r\nL 79.890625 0 \r\nL 79.890625 32.71875 \r\nQ 79.890625 40.578125 77.09375 44.375 \r\nQ 74.3125 48.1875 68.609375 48.1875 \r\nQ 61.625 48.1875 57.5625 43.546875 \r\nQ 53.515625 38.921875 53.515625 30.90625 \r\nL 53.515625 0 \r\nL 44.484375 0 \r\nL 44.484375 32.71875 \r\nQ 44.484375 40.625 41.703125 44.40625 \r\nQ 38.921875 48.1875 33.109375 48.1875 \r\nQ 26.21875 48.1875 22.15625 43.53125 \r\nQ 18.109375 38.875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.1875 51.21875 25.484375 53.609375 \r\nQ 29.78125 56 35.6875 56 \r\nQ 41.65625 56 45.828125 52.96875 \r\nQ 50 49.953125 52 44.1875 \r\nz\r\n\" id=\"DejaVuSans-109\"/>\r\n      <path d=\"M 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\nM 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 75.984375 \r\nL 18.109375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-98\"/>\r\n      <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n      <path d=\"M 41.109375 46.296875 \r\nQ 39.59375 47.171875 37.8125 47.578125 \r\nQ 36.03125 48 33.890625 48 \r\nQ 26.265625 48 22.1875 43.046875 \r\nQ 18.109375 38.09375 18.109375 28.8125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 20.953125 51.171875 25.484375 53.578125 \r\nQ 30.03125 56 36.53125 56 \r\nQ 37.453125 56 38.578125 55.875 \r\nQ 39.703125 55.765625 41.0625 55.515625 \r\nz\r\n\" id=\"DejaVuSans-114\"/>\r\n      <path id=\"DejaVuSans-32\"/>\r\n      <path d=\"M 30.609375 48.390625 \r\nQ 23.390625 48.390625 19.1875 42.75 \r\nQ 14.984375 37.109375 14.984375 27.296875 \r\nQ 14.984375 17.484375 19.15625 11.84375 \r\nQ 23.34375 6.203125 30.609375 6.203125 \r\nQ 37.796875 6.203125 41.984375 11.859375 \r\nQ 46.1875 17.53125 46.1875 27.296875 \r\nQ 46.1875 37.015625 41.984375 42.703125 \r\nQ 37.796875 48.390625 30.609375 48.390625 \r\nz\r\nM 30.609375 56 \r\nQ 42.328125 56 49.015625 48.375 \r\nQ 55.71875 40.765625 55.71875 27.296875 \r\nQ 55.71875 13.875 49.015625 6.21875 \r\nQ 42.328125 -1.421875 30.609375 -1.421875 \r\nQ 18.84375 -1.421875 12.171875 6.21875 \r\nQ 5.515625 13.875 5.515625 27.296875 \r\nQ 5.515625 40.765625 12.171875 48.375 \r\nQ 18.84375 56 30.609375 56 \r\nz\r\n\" id=\"DejaVuSans-111\"/>\r\n      <path d=\"M 37.109375 75.984375 \r\nL 37.109375 68.5 \r\nL 28.515625 68.5 \r\nQ 23.6875 68.5 21.796875 66.546875 \r\nQ 19.921875 64.59375 19.921875 59.515625 \r\nL 19.921875 54.6875 \r\nL 34.71875 54.6875 \r\nL 34.71875 47.703125 \r\nL 19.921875 47.703125 \r\nL 19.921875 0 \r\nL 10.890625 0 \r\nL 10.890625 47.703125 \r\nL 2.296875 47.703125 \r\nL 2.296875 54.6875 \r\nL 10.890625 54.6875 \r\nL 10.890625 58.5 \r\nQ 10.890625 67.625 15.140625 71.796875 \r\nQ 19.390625 75.984375 28.609375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-102\"/>\r\n      <path d=\"M 19.671875 64.796875 \r\nL 19.671875 37.40625 \r\nL 32.078125 37.40625 \r\nQ 38.96875 37.40625 42.71875 40.96875 \r\nQ 46.484375 44.53125 46.484375 51.125 \r\nQ 46.484375 57.671875 42.71875 61.234375 \r\nQ 38.96875 64.796875 32.078125 64.796875 \r\nz\r\nM 9.8125 72.90625 \r\nL 32.078125 72.90625 \r\nQ 44.34375 72.90625 50.609375 67.359375 \r\nQ 56.890625 61.8125 56.890625 51.125 \r\nQ 56.890625 40.328125 50.609375 34.8125 \r\nQ 44.34375 29.296875 32.078125 29.296875 \r\nL 19.671875 29.296875 \r\nL 19.671875 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-80\"/>\r\n      <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-105\"/>\r\n      <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n      <path d=\"M 48.78125 52.59375 \r\nL 48.78125 44.1875 \r\nQ 44.96875 46.296875 41.140625 47.34375 \r\nQ 37.3125 48.390625 33.40625 48.390625 \r\nQ 24.65625 48.390625 19.8125 42.84375 \r\nQ 14.984375 37.3125 14.984375 27.296875 \r\nQ 14.984375 17.28125 19.8125 11.734375 \r\nQ 24.65625 6.203125 33.40625 6.203125 \r\nQ 37.3125 6.203125 41.140625 7.25 \r\nQ 44.96875 8.296875 48.78125 10.40625 \r\nL 48.78125 2.09375 \r\nQ 45.015625 0.34375 40.984375 -0.53125 \r\nQ 36.96875 -1.421875 32.421875 -1.421875 \r\nQ 20.0625 -1.421875 12.78125 6.34375 \r\nQ 5.515625 14.109375 5.515625 27.296875 \r\nQ 5.515625 40.671875 12.859375 48.328125 \r\nQ 20.21875 56 33.015625 56 \r\nQ 37.15625 56 41.109375 55.140625 \r\nQ 45.0625 54.296875 48.78125 52.59375 \r\nz\r\n\" id=\"DejaVuSans-99\"/>\r\n      <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-112\"/>\r\n      <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n      <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-108\"/>\r\n      <path d=\"M 64.40625 67.28125 \r\nL 64.40625 56.890625 \r\nQ 59.421875 61.53125 53.78125 63.8125 \r\nQ 48.140625 66.109375 41.796875 66.109375 \r\nQ 29.296875 66.109375 22.65625 58.46875 \r\nQ 16.015625 50.828125 16.015625 36.375 \r\nQ 16.015625 21.96875 22.65625 14.328125 \r\nQ 29.296875 6.6875 41.796875 6.6875 \r\nQ 48.140625 6.6875 53.78125 8.984375 \r\nQ 59.421875 11.28125 64.40625 15.921875 \r\nL 64.40625 5.609375 \r\nQ 59.234375 2.09375 53.4375 0.328125 \r\nQ 47.65625 -1.421875 41.21875 -1.421875 \r\nQ 24.65625 -1.421875 15.125 8.703125 \r\nQ 5.609375 18.84375 5.609375 36.375 \r\nQ 5.609375 53.953125 15.125 64.078125 \r\nQ 24.65625 74.21875 41.21875 74.21875 \r\nQ 47.75 74.21875 53.53125 72.484375 \r\nQ 59.328125 70.75 64.40625 67.28125 \r\nz\r\n\" id=\"DejaVuSans-67\"/>\r\n      <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n      <path d=\"M 44.28125 53.078125 \r\nL 44.28125 44.578125 \r\nQ 40.484375 46.53125 36.375 47.5 \r\nQ 32.28125 48.484375 27.875 48.484375 \r\nQ 21.1875 48.484375 17.84375 46.4375 \r\nQ 14.5 44.390625 14.5 40.28125 \r\nQ 14.5 37.15625 16.890625 35.375 \r\nQ 19.28125 33.59375 26.515625 31.984375 \r\nL 29.59375 31.296875 \r\nQ 39.15625 29.25 43.1875 25.515625 \r\nQ 47.21875 21.78125 47.21875 15.09375 \r\nQ 47.21875 7.46875 41.1875 3.015625 \r\nQ 35.15625 -1.421875 24.609375 -1.421875 \r\nQ 20.21875 -1.421875 15.453125 -0.5625 \r\nQ 10.6875 0.296875 5.421875 2 \r\nL 5.421875 11.28125 \r\nQ 10.40625 8.6875 15.234375 7.390625 \r\nQ 20.0625 6.109375 24.8125 6.109375 \r\nQ 31.15625 6.109375 34.5625 8.28125 \r\nQ 37.984375 10.453125 37.984375 14.40625 \r\nQ 37.984375 18.0625 35.515625 20.015625 \r\nQ 33.0625 21.96875 24.703125 23.78125 \r\nL 21.578125 24.515625 \r\nQ 13.234375 26.265625 9.515625 29.90625 \r\nQ 5.8125 33.546875 5.8125 39.890625 \r\nQ 5.8125 47.609375 11.28125 51.796875 \r\nQ 16.75 56 26.8125 56 \r\nQ 31.78125 56 36.171875 55.265625 \r\nQ 40.578125 54.546875 44.28125 53.078125 \r\nz\r\n\" id=\"DejaVuSans-115\"/>\r\n     </defs>\r\n     <g transform=\"translate(134.93125 252.916563)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-78\"/>\r\n      <use x=\"74.804688\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"138.183594\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"235.595703\" xlink:href=\"#DejaVuSans-98\"/>\r\n      <use x=\"299.072266\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"360.595703\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"401.708984\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"433.496094\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"494.677734\" xlink:href=\"#DejaVuSans-102\"/>\r\n      <use x=\"529.882812\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"561.669922\" xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"620.222656\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"661.335938\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"689.119141\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"752.498047\" xlink:href=\"#DejaVuSans-99\"/>\r\n      <use x=\"807.478516\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"835.261719\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"898.738281\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"960.017578\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"987.800781\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"1019.587891\" xlink:href=\"#DejaVuSans-67\"/>\r\n      <use x=\"1089.412109\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"1150.59375\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"1248.005859\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"1311.482422\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"1372.664062\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1436.042969\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"1497.566406\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1560.945312\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"1600.154297\" xlink:href=\"#DejaVuSans-115\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_10\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"mc06294e92f\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"211.199945\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 0.45 -->\r\n      <g transform=\"translate(20.878125 214.999164)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_11\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"186.773136\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_12\">\r\n      <!-- 0.50 -->\r\n      <g transform=\"translate(20.878125 190.572355)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"162.346327\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_13\">\r\n      <!-- 0.55 -->\r\n      <g transform=\"translate(20.878125 166.145545)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_13\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"137.919517\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_14\">\r\n      <!-- 0.60 -->\r\n      <defs>\r\n       <path d=\"M 33.015625 40.375 \r\nQ 26.375 40.375 22.484375 35.828125 \r\nQ 18.609375 31.296875 18.609375 23.390625 \r\nQ 18.609375 15.53125 22.484375 10.953125 \r\nQ 26.375 6.390625 33.015625 6.390625 \r\nQ 39.65625 6.390625 43.53125 10.953125 \r\nQ 47.40625 15.53125 47.40625 23.390625 \r\nQ 47.40625 31.296875 43.53125 35.828125 \r\nQ 39.65625 40.375 33.015625 40.375 \r\nz\r\nM 52.59375 71.296875 \r\nL 52.59375 62.3125 \r\nQ 48.875 64.0625 45.09375 64.984375 \r\nQ 41.3125 65.921875 37.59375 65.921875 \r\nQ 27.828125 65.921875 22.671875 59.328125 \r\nQ 17.53125 52.734375 16.796875 39.40625 \r\nQ 19.671875 43.65625 24.015625 45.921875 \r\nQ 28.375 48.1875 33.59375 48.1875 \r\nQ 44.578125 48.1875 50.953125 41.515625 \r\nQ 57.328125 34.859375 57.328125 23.390625 \r\nQ 57.328125 12.15625 50.6875 5.359375 \r\nQ 44.046875 -1.421875 33.015625 -1.421875 \r\nQ 20.359375 -1.421875 13.671875 8.265625 \r\nQ 6.984375 17.96875 6.984375 36.375 \r\nQ 6.984375 53.65625 15.1875 63.9375 \r\nQ 23.390625 74.21875 37.203125 74.21875 \r\nQ 40.921875 74.21875 44.703125 73.484375 \r\nQ 48.484375 72.75 52.59375 71.296875 \r\nz\r\n\" id=\"DejaVuSans-54\"/>\r\n      </defs>\r\n      <g transform=\"translate(20.878125 141.718736)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"113.492708\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_15\">\r\n      <!-- 0.65 -->\r\n      <g transform=\"translate(20.878125 117.291927)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_6\">\r\n     <g id=\"line2d_15\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"89.065898\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_16\">\r\n      <!-- 0.70 -->\r\n      <defs>\r\n       <path d=\"M 8.203125 72.90625 \r\nL 55.078125 72.90625 \r\nL 55.078125 68.703125 \r\nL 28.609375 0 \r\nL 18.3125 0 \r\nL 43.21875 64.59375 \r\nL 8.203125 64.59375 \r\nz\r\n\" id=\"DejaVuSans-55\"/>\r\n      </defs>\r\n      <g transform=\"translate(20.878125 92.865117)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-55\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_7\">\r\n     <g id=\"line2d_16\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"64.639089\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_17\">\r\n      <!-- 0.75 -->\r\n      <g transform=\"translate(20.878125 68.438308)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-55\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_8\">\r\n     <g id=\"line2d_17\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"40.21228\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_18\">\r\n      <!-- 0.80 -->\r\n      <defs>\r\n       <path d=\"M 31.78125 34.625 \r\nQ 24.75 34.625 20.71875 30.859375 \r\nQ 16.703125 27.09375 16.703125 20.515625 \r\nQ 16.703125 13.921875 20.71875 10.15625 \r\nQ 24.75 6.390625 31.78125 6.390625 \r\nQ 38.8125 6.390625 42.859375 10.171875 \r\nQ 46.921875 13.96875 46.921875 20.515625 \r\nQ 46.921875 27.09375 42.890625 30.859375 \r\nQ 38.875 34.625 31.78125 34.625 \r\nz\r\nM 21.921875 38.8125 \r\nQ 15.578125 40.375 12.03125 44.71875 \r\nQ 8.5 49.078125 8.5 55.328125 \r\nQ 8.5 64.0625 14.71875 69.140625 \r\nQ 20.953125 74.21875 31.78125 74.21875 \r\nQ 42.671875 74.21875 48.875 69.140625 \r\nQ 55.078125 64.0625 55.078125 55.328125 \r\nQ 55.078125 49.078125 51.53125 44.71875 \r\nQ 48 40.375 41.703125 38.8125 \r\nQ 48.828125 37.15625 52.796875 32.3125 \r\nQ 56.78125 27.484375 56.78125 20.515625 \r\nQ 56.78125 9.90625 50.3125 4.234375 \r\nQ 43.84375 -1.421875 31.78125 -1.421875 \r\nQ 19.734375 -1.421875 13.25 4.234375 \r\nQ 6.78125 9.90625 6.78125 20.515625 \r\nQ 6.78125 27.484375 10.78125 32.3125 \r\nQ 14.796875 37.15625 21.921875 38.8125 \r\nz\r\nM 18.3125 54.390625 \r\nQ 18.3125 48.734375 21.84375 45.5625 \r\nQ 25.390625 42.390625 31.78125 42.390625 \r\nQ 38.140625 42.390625 41.71875 45.5625 \r\nQ 45.3125 48.734375 45.3125 54.390625 \r\nQ 45.3125 60.0625 41.71875 63.234375 \r\nQ 38.140625 66.40625 31.78125 66.40625 \r\nQ 25.390625 66.40625 21.84375 63.234375 \r\nQ 18.3125 60.0625 18.3125 54.390625 \r\nz\r\n\" id=\"DejaVuSans-56\"/>\r\n      </defs>\r\n      <g transform=\"translate(20.878125 44.011498)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_9\">\r\n     <g id=\"line2d_18\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"50.14375\" xlink:href=\"#mc06294e92f\" y=\"15.78547\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_19\">\r\n      <!-- 0.85 -->\r\n      <g transform=\"translate(20.878125 19.584689)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_20\">\r\n     <!-- Cummulative Explained Variance -->\r\n     <defs>\r\n      <path d=\"M 2.984375 54.6875 \r\nL 12.5 54.6875 \r\nL 29.59375 8.796875 \r\nL 46.6875 54.6875 \r\nL 56.203125 54.6875 \r\nL 35.6875 0 \r\nL 23.484375 0 \r\nz\r\n\" id=\"DejaVuSans-118\"/>\r\n      <path d=\"M 9.8125 72.90625 \r\nL 55.90625 72.90625 \r\nL 55.90625 64.59375 \r\nL 19.671875 64.59375 \r\nL 19.671875 43.015625 \r\nL 54.390625 43.015625 \r\nL 54.390625 34.71875 \r\nL 19.671875 34.71875 \r\nL 19.671875 8.296875 \r\nL 56.78125 8.296875 \r\nL 56.78125 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-69\"/>\r\n      <path d=\"M 54.890625 54.6875 \r\nL 35.109375 28.078125 \r\nL 55.90625 0 \r\nL 45.3125 0 \r\nL 29.390625 21.484375 \r\nL 13.484375 0 \r\nL 2.875 0 \r\nL 24.125 28.609375 \r\nL 4.6875 54.6875 \r\nL 15.28125 54.6875 \r\nL 29.78125 35.203125 \r\nL 44.28125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-120\"/>\r\n      <path d=\"M 45.40625 46.390625 \r\nL 45.40625 75.984375 \r\nL 54.390625 75.984375 \r\nL 54.390625 0 \r\nL 45.40625 0 \r\nL 45.40625 8.203125 \r\nQ 42.578125 3.328125 38.25 0.953125 \r\nQ 33.9375 -1.421875 27.875 -1.421875 \r\nQ 17.96875 -1.421875 11.734375 6.484375 \r\nQ 5.515625 14.40625 5.515625 27.296875 \r\nQ 5.515625 40.1875 11.734375 48.09375 \r\nQ 17.96875 56 27.875 56 \r\nQ 33.9375 56 38.25 53.625 \r\nQ 42.578125 51.265625 45.40625 46.390625 \r\nz\r\nM 14.796875 27.296875 \r\nQ 14.796875 17.390625 18.875 11.75 \r\nQ 22.953125 6.109375 30.078125 6.109375 \r\nQ 37.203125 6.109375 41.296875 11.75 \r\nQ 45.40625 17.390625 45.40625 27.296875 \r\nQ 45.40625 37.203125 41.296875 42.84375 \r\nQ 37.203125 48.484375 30.078125 48.484375 \r\nQ 22.953125 48.484375 18.875 42.84375 \r\nQ 14.796875 37.203125 14.796875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-100\"/>\r\n      <path d=\"M 28.609375 0 \r\nL 0.78125 72.90625 \r\nL 11.078125 72.90625 \r\nL 34.1875 11.53125 \r\nL 57.328125 72.90625 \r\nL 67.578125 72.90625 \r\nL 39.796875 0 \r\nz\r\n\" id=\"DejaVuSans-86\"/>\r\n     </defs>\r\n     <g transform=\"translate(14.798438 198.660625)rotate(-90)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-67\"/>\r\n      <use x=\"69.824219\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"133.203125\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"230.615234\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"328.027344\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"391.40625\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"419.189453\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"480.46875\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"519.677734\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"547.460938\" xlink:href=\"#DejaVuSans-118\"/>\r\n      <use x=\"606.640625\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"668.164062\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"699.951172\" xlink:href=\"#DejaVuSans-69\"/>\r\n      <use x=\"763.134766\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"822.314453\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"885.791016\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"913.574219\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"974.853516\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"1002.636719\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1066.015625\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"1127.539062\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"1191.015625\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"1222.802734\" xlink:href=\"#DejaVuSans-86\"/>\r\n      <use x=\"1283.460938\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"1344.740234\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"1385.853516\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"1413.636719\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"1474.916016\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1538.294922\" xlink:href=\"#DejaVuSans-99\"/>\r\n      <use x=\"1593.275391\" xlink:href=\"#DejaVuSans-101\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_19\">\r\n    <path clip-path=\"url(#p2b8ca599fe)\" d=\"M 65.361932 214.756364 \r\nL 141.452841 122.075274 \r\nL 217.54375 76.186272 \r\nL 293.634659 43.932454 \r\nL 369.725568 17.083636 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 50.14375 224.64 \r\nL 50.14375 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 384.94375 224.64 \r\nL 384.94375 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 50.14375 224.64 \r\nL 384.94375 224.64 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 50.14375 7.2 \r\nL 384.94375 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p2b8ca599fe\">\r\n   <rect height=\"217.44\" width=\"334.8\" x=\"50.14375\" y=\"7.2\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "source": [
    "plt.plot(np.cumsum(pca.explained_variance_ratio_))\n",
    "plt.xlabel(\"Number of Principal Components\")\n",
    "plt.ylabel(\"Cummulative Explained Variance\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ]
}