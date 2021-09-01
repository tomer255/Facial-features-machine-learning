# Facial features machine learning 

## Main objective 
### identify a gender of person by his facial features

---------------------------------------------------------------------
### Recommend to check the facial features machine learning .pptx for project breakdown

---------------------------------------------------------------------
## step 1 : Scraping 
#### crawling.py
Scraping images from shutterstock.com website in order to fill our dataset with labeled images by gender.
eventually our database contained over 70,000 labeled images.


## step 2 : Data 

#### shape_predictor_68_face_landmarks.dat
We used shape_predictor_68_face_landmarks file which detecting a face in an image and setting 68 points on the face in specific locations.

#### create_df.py
```py
def create_DataFrame():
    num_face_landmarks = 68
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_"+str(num_face_landmarks)+"_face_landmarks.dat")
    df = pd.DataFrame()
    for n in range(0, num_face_landmarks):
        df['point_' + str(n+1)] = 0
    df['Age'] = 0
    df['is_male'] = 0
    images_paths = os.listdir('./images')
    for path in images_paths:
        print(path)
        path = './images/'+path
        dict_arry = image_to_dict(detector,predictor,path,num_face_landmarks)
        for dict in dict_arry:
            df = df.append(dict, ignore_index=True)
    df.to_csv("face_landmarks.csv")
```
In that part of the project we created our dataframe and we stored each coordinate of points that our shape_predictor_68_face_landmarks recognized ,
each column of the dataframe were the location of the point on the face.

#### Data_handeling.py
After we had a dataframe full of coordiantes for each point we converted our X and Y coordinates to specific distances between points in order to represent facial features on the face.
```py
new_df = pd.DataFrame()
    for num, points in enumerate(points_arr):
        s1x = df["point_" + str(points[0] + 1) + '_x']
        s1y = df["point_" + str(points[0] + 1) + '_y']
        s2x = df["point_" + str(points[1] + 1) + '_x']
        s2y = df["point_" + str(points[1] + 1) + '_y']
        new_df["distance" + '_' + str(points[0]) + '_' + str(points[1])] = numpy.sqrt(
            ((s1x - s2x) ** 2) + ((s1y - s2y) ** 2))
    new_df['Age'] = df['Age']
    new_df['is_female'] = df['is_female']

    new_df.to_csv("vectored_data.csv")
    return new_df
```

## step 3 : Model
#### build_model.py
In this section of the project we looked for a model which will give us the best accuracy on our data,
after we tried a myriad of models the one who gave us the best accuracy was SVM .
```py
def SVM2(X_train, y_train):
    param_grid = {'C': [1e3,1e4,1e5],
                  'gamma': [0.0001,0.0005,0.00001], }
    clf = GridSearchCV(
        SVC(kernel='linear', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train, y_train)
    return clf
```


## step 4 : Interpretation
Our conclusions about the project were based on the accuracy of the model.
```py
SVM :
{'C': 1000.0, 'gamma': 0.0001}
train score:
[[8781  873]
 [ 702 8204]]
     accuracy is: 0.9151400862068966
     precision is: 0.9038228489589071
     recall is: 0.9211767347855379
     f1 is: 0.9124172829894901
test score:
[[2098  352]
 [ 326 1864]]
     accuracy is: 0.8538793103448276
     precision is: 0.8411552346570397
     recall is: 0.8511415525114155
     f1 is: 0.8461189287335452
```

* We found out that it is possible to recognize gender by facial features.

* We found that there is a difference in specific facial features between male and female for example the size of the face in males images was bigger than the females ,and the lips size of the females was bigger than the males as we mentioned earlier.

* For summary we enjoyed doing that project ,the project was challenging, and we had to learn new things and expand horizons in order to answer the main question of the project.

