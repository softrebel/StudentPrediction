from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data = pd.read_csv('ArticleDataset.csv')

y, X = dmatrices('Class ~ raisedhands + VisITedResources + AnnouncementsView + Discussion + \
                  ParentschoolSatisfaction + StudentAbsenceDays ',
                  data, return_type="dataframe")

ny = y[['Class[H]']]

X_train, X_test, y_train, y_test = train_test_split(X, ny, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
# generate evaluation metrics
print (metrics.accuracy_score(y_test, predicted))