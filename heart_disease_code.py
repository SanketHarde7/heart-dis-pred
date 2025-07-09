from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df=pd.read_csv(r"C:\Users\sanke\Downloads\heart.csv")


df['Sex'] = df['Sex'].replace({'M': 1, 'F': 0}).astype(int)
df['ExerciseAngina'] = df['ExerciseAngina'].replace({'Y': 1, 'N': 0})
df['ChestPainType'] = df['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
df['RestingECG'] = df['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ST_Slope'] = df['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})

x = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


print("classification report :\n",classification_report(y_pred,y_test))
print("accuracy score ",accuracy_score(y_pred,y_test))

print(df.head())
joblib.dump(model,'heart_dis.pkl')


