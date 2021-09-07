{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from sklearn.datasets import load_iris\
from sklearn.model_selection import cross_val_score\
from sklearn.tree import DecisionTreeClassifier\
import pickle\
\pard\pardeftab720\sl320\partightenfactor0

\f1\fs26 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import pandas as pd\
\
def make_prediction(model, data):\
    x = model.predict(data)\
    df = pd.DataFrame(iris.data)\
    df['actual'] = iris.target\
    df['pred'] = x\
    df.to_csv('iris_pred.csv')\
\
}