from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

print("\nTraining complete. Now you can use gender predictor.\n")
height=input("Height (in cm)(180): ")
height=int(height)
weight=input("Weight (in kg)(70): ")
weight=int(weight)
shoe_size=input("Shoe Size (40): ")
shoe_size=int(shoe_size)

prediction = clf.predict([[height, weight, shoe_size]])

# CHALLENGE compare their results and print the best one!
print(prediction)
print("Predicted Gender is:-- " + prediction[0] + " --")