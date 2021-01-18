import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense


missing_values = ["n/a", "na", "-", "?"]
factors = [
           'id',
           'clump_thickness',
           'uniformity_of_cell_size',
           'uniformity_of_cell_shape',
           'marginal_adhension',
           'single_epithelial_cell_size',
           'bare_nuclei',
           'bland_chromatin',
           'normal_nucleoli',
           'mitoses',
           'class'
]

data = pd.read_csv("breast-cancer-wisconsin.data", na_values=missing_values, names=factors)
data = data.drop(['id'], axis=1)

# Take a look at the first few rows
print(data.head(10))


print("\nBaza przed uzupełnieniem brakujących danych.\n")
print(data.isnull().sum())

#Find a percentage of missing values in each column.

percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_data = pd.DataFrame({'percent_missing': percent_missing})
print("\n"+ str(missing_value_data) + "\n")

labels = 'Missing Data from column', 'Correct Data in column'
missing_data_value = missing_value_data.values[5][0]

sizes = [missing_data_value, 100.0 - missing_data_value]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# Replace missing values with column medians

min_max_mean_classes = ['clump_thickness',
           'uniformity_of_cell_size',
           'uniformity_of_cell_shape',
           'marginal_adhension',
           'single_epithelial_cell_size',
           'bare_nuclei',
           'bland_chromatin',
           'normal_nucleoli',
           'mitoses']

min = data[min_max_mean_classes].min().values
max = data[min_max_mean_classes].max().values
mean = data[min_max_mean_classes].mean().values

barWidth = 0.25
r1 = np.arange(len(min))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, min, color='#A1AB4F', width=barWidth, edgecolor='white', label='min')
plt.bar(r2, max, color='#794420', width=barWidth, edgecolor='white', label='max')
plt.bar(r3, mean, color='#6F686F', width=barWidth, edgecolor='white', label='mean')

plt.xlabel('Columns', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(min))], min_max_mean_classes)
plt.legend()
plt.show()

plt.rcParams["figure.figsize"] = [25,9]
plt.show()



for (columnName, columnData) in data.iteritems():
    if columnName != 'class':
        median = data[columnName].median()
        data[columnName].fillna(median, inplace=True)

print("\nBaza po uzupełnieniu medianą kolumny brakujących danych.\n")
print(str(data.isnull().sum()) + '\n\n')

train, test = train_test_split(data, test_size=0.3, random_state=50)

def prepare_knn_cls(data, neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(*data_fit(data))
    return knn


def prepare_tree_cls(data):
    tree = DecisionTreeClassifier()
    tree.fit(*data_fit(data))
    return tree


def prepare_nb_cls(data):
    nb = GaussianNB()
    nb.fit(*data_fit(data))
    return nb


def data_fit(d):
    return d.drop('class', axis=1), d['class']


def evaluate(cls, data):
    return cls.score(*data_fit(data))


def cf_matrix(cls, data):
    cf = confusion_matrix(data['class'], cls.predict(data_fit(data)[0]))
    return cf


nn3 = prepare_knn_cls(train, 3)
nn5 = prepare_knn_cls(train, 5)
nn11 = prepare_knn_cls(train, 11)
tree = prepare_tree_cls(train)
nb = prepare_nb_cls(train)
cls_names = ['3nn', '5nn', '11nn', 'tree', 'nb']

for cls, name in zip([nn3, nn5, nn11, tree, nb], cls_names):
    summary = f'{name} Accuracy = {(evaluate(cls, test)*100):.6f} %\nConfusion Matrix = \n{cf_matrix(cls, test)}\n'
    print(summary)

evaluations = [evaluate(cls, test)*100 for cls in [nn3, nn5, nn11, tree, nb]]
plt.bar(cls_names, evaluations, align='center', width=barWidth, color='red')
plt.ylabel("Percentage of accuracy")
plt.show()


###############
#SIECI NEURONOWE

def map_result (value):
    if value == 2.0:
        return 1, 0
    if value == 4.0:
        return 0, 1
    return -1, -1


class_names_for_normalization = ['clump_thickness',
           'uniformity_of_cell_size',
           'uniformity_of_cell_shape',
           'marginal_adhension',
           'single_epithelial_cell_size',
           'bare_nuclei',
           'bland_chromatin',
           'normal_nucleoli',
           'mitoses']

results = ['bening', 'malignant']
normalization = lambda x: (x - x.min()) / (x.max() - x.min())
data[class_names_for_normalization] = data[class_names_for_normalization].apply(normalization)

bening = []
malignant = []
for index, row in data.iterrows():
    result = map_result(row['class'])
    bening.append(result[0])
    malignant.append(result[1])

data['bening'] = bening
data['malignant'] = malignant

train = data.sample(frac=0.7)
test = data[~data.isin(train).all(1)]
print(data)
# print(train)
# print(test)

model = Sequential()
model.add(Dense(10, input_shape=(9,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(2, activation='softmax', name='output'))
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# print(train[class_names_for_normalization])
# print(train[results])
model.fit(train[class_names_for_normalization], train[results], epochs=50, verbose=2, batch_size=10)
result = model.evaluate(test[class_names_for_normalization], test[results])

pred = model.predict(test[class_names_for_normalization])
pred = (pred > 0.5).astype(int)
conf_matrix = multilabel_confusion_matrix(test[results], pred)
print(conf_matrix)
print(f"acc = {result[1]:.2f}")
print(f"loss = {result[0]:.4f}")