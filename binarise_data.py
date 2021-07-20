import numpy as np
import pandas as pd

def zoo():
    # Zoo 101 x 17

    # original data of shape (101, 18) with classification column and unique animal name column
    zoo_categorical_w_id_and_classification = np.loadtxt('./data/raw_data/Zoo/zoo.data.txt', delimiter=',',dtype=object)

    # the first column is the unique animal name for each instance
    # the last column is the classification attribute
    # we drop these columns to get (101, 16)
    zoo_categorical = zoo_categorical_w_id_and_classification[:, 1:-1]

    # legs		Numeric (set of values: {0,2,4,5,6,8})
    # replace the categorical column wtih two Boolean columns
    # 1. true if legs <=4
    zoo_legs_less_than_5 = np.zeros([101, 1],dtype=int)
    zoo_legs_less_than_5[zoo_categorical[:, 12] == '0'] = 1
    zoo_legs_less_than_5[zoo_categorical[:, 12] == '2'] = 1
    zoo_legs_less_than_5[zoo_categorical[:, 12] == '4'] = 1
    # 1. true if legs >=5
    zoo_legs_more_than_4 = np.zeros([101, 1],dtype=int)
    zoo_legs_more_than_4[zoo_categorical[:, 12] == '5'] = 1
    zoo_legs_more_than_4[zoo_categorical[:, 12] == '6'] = 1
    zoo_legs_more_than_4[zoo_categorical[:, 12] == '8'] = 1

    # the rest of the columns is Boolean
    zoo_Boolean = zoo_categorical[:, np.arange(16) != 12 ]

    # we have no missing values
    zoo_Boolean[zoo_Boolean == '0'] = 0
    zoo_Boolean[zoo_Boolean == '1'] = 1

    zoo_Boolean = zoo_Boolean.astype(int)

    zoo = np.concatenate((zoo_Boolean,zoo_legs_less_than_5, zoo_legs_more_than_4),axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(zoo == 0, axis=1)
    zoo = zoo[~is_zero_row, :]
    is_zero_col = np.all(zoo == 0, axis=0)
    zoo = zoo[:, ~is_zero_col]


    np.savetxt('./data/zoo.txt', zoo, fmt='%u')

def tumor():
    #PRIMARY TUMOR, 339 x 24

    #original data of shape (339, 18) with classification column (1st col) and some non-binary variables
    tumor_categorical_w_classification = np.loadtxt('./data/raw_data/Tumor/primary-tumor.data.txt', delimiter=',', dtype = object)
    # shape (339, 17)
    tumor_categorical = tumor_categorical_w_classification[:,1:]

    #the first 4 columns are non-binary caterogical variables

    # age:   <30, 30-59, >=60
    age30 = np.zeros([339,1],dtype=int)
    age30[tumor_categorical[:,0] == '1'] = 1
    age30_59 = np.zeros([339,1],dtype=int)
    age30_59[tumor_categorical[:,0] == '2'] = 1
    age60 = np.zeros([339,1],dtype=int)
    age60[tumor_categorical[:,0] == '3'] = 1
    # sex:   male, female
    sex_male = np.zeros([339,1],dtype=int)
    sex_male[tumor_categorical[:,1] == '1'] = 1
    sex_female = np.zeros([339,1],dtype=int)
    sex_female[tumor_categorical[:,1] == '2'] = 1
    # histologic-type: epidermoid, adeno, anaplastic
    hist1 = np.zeros([339,1],dtype=int)
    hist1[tumor_categorical[:,2] == '1'] = 1
    hist2 = np.zeros([339,1],dtype=int)
    hist2[tumor_categorical[:,2] == '2'] = 1
    hist3 = np.zeros([339,1],dtype=int)
    hist3[tumor_categorical[:,2] == '3'] = 1
    # degree-of-diffe: well, fairly, poorly
    deg1 = np.zeros([339,1],dtype=int)
    deg1[tumor_categorical[:,3] == '1'] = 1
    deg2 = np.zeros([339,1],dtype=int)
    deg2[tumor_categorical[:,3] == '2'] = 1
    deg3 = np.zeros([339,1],dtype=int)
    deg3[tumor_categorical[:,3] == '3'] = 1
    tumor_non_binary_part = np.concatenate((age30,age30_59,age60,
                                            sex_male,sex_female,
                                            hist1,hist2,hist3,
                                            deg1,deg2,deg3), axis=1)

    # the rest of the columns is binary, but there are 2 missing values
    tumor_binary_part = tumor_categorical[:, 4:]
    # convert yes to 1
    tumor_binary_part[tumor_binary_part == '1'] = '1'
    #convert no to 0
    tumor_binary_part[tumor_binary_part == '2'] = '0'

    # np.where(tumor_binary_part == '?')
    # (array([9, 16]), array([7, 10]))
    # WITHOUT ANY JUSTIFICATION WE ASSIGN 0 VALUE TO THE 2 MISSING ENTRIES IN THE BINARY COLUMNS
    tumor_binary_part[tumor_binary_part == '?'] = '0'


    tumor_binary_part = tumor_binary_part.astype(int)

    # shape (339, 24)
    tumor = np.concatenate((tumor_non_binary_part,tumor_binary_part), axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(tumor == 0, axis=1)
    tumor = tumor[~is_zero_row, :]
    is_zero_col = np.all(tumor == 0, axis=0)
    # shape (339, 24)
    tumor = tumor[:, ~is_zero_col]

    np.savetxt('./data/tumor.txt', tumor, fmt='%u')

def tumor_w_missing():
    #PRIMARY TUMOR, 339 x 24

    #original data of shape (339, 18) with classification column (1st col) and some non-binary variables
    tumor_categorical_w_classification = np.loadtxt('./data/raw_data/Tumor/primary-tumor.data.txt', delimiter=',', dtype = object)
    # shape (339, 17)
    tumor_categorical = tumor_categorical_w_classification[:,1:]

    #the first 4 columns are non-binary caterogical variables

    # age:   <30, 30-59, >=60.   no missing values
    age30 = np.zeros([339,1],dtype=int)
    age30[tumor_categorical[:,0] == '1'] = 1
    age30_59 = np.zeros([339,1],dtype=int)
    age30_59[tumor_categorical[:,0] == '2'] = 1
    age60 = np.zeros([339,1],dtype=int)
    age60[tumor_categorical[:,0] == '3'] = 1

    # sex:   male, female.      1 missing value
    sex_male = np.zeros([339,1],dtype=float)
    sex_male[tumor_categorical[:, 1] == '1'] = 1
    sex_male[tumor_categorical[:, 1] == '?'] = np.nan
    sex_female = np.zeros([339,1],dtype=float)
    sex_female[tumor_categorical[:, 1] == '2'] = 1
    sex_female[tumor_categorical[:, 1] == '?'] = np.nan

    # histologic-type: epidermoid, adeno, anaplastic.    67 missing values
    hist1 = np.zeros([339,1],dtype=float)
    hist1[tumor_categorical[:, 2] == '1'] = 1
    hist1[tumor_categorical[:, 2] == '?'] = np.nan
    hist2 = np.zeros([339,1],dtype=float)
    hist2[tumor_categorical[:, 2] == '2'] = 1
    hist2[tumor_categorical[:, 2] == '?'] = np.nan
    hist3 = np.zeros([339,1],dtype=float)
    hist3[tumor_categorical[:, 2] == '3'] = 1
    hist3[tumor_categorical[:, 2] == '?'] = np.nan

    # degree-of-diffe: well, fairly, poorly.    155 missing values
    deg1 = np.zeros([339,1],dtype=float)
    deg1[tumor_categorical[:, 3] == '1'] = 1
    deg1[tumor_categorical[:, 3] == '?'] = np.nan
    deg2 = np.zeros([339,1],dtype=float)
    deg2[tumor_categorical[:, 3] == '2'] = 1
    deg2[tumor_categorical[:, 3] == '?'] = np.nan
    deg3 = np.zeros([339,1],dtype=float)
    deg3[tumor_categorical[:, 3] == '3'] = 1
    deg3[tumor_categorical[:, 3] == '?'] = np.nan

    tumor_non_binary_part = np.concatenate((age30,age30_59,age60,
                                            sex_male,sex_female,
                                            hist1,hist2,hist3,
                                            deg1,deg2,deg3), axis=1)

    # the rest of the columns is binary, but there are 2 missing values
    tumor_binary_part = tumor_categorical[:, 4:]
    # convert yes to 1
    tumor_binary_part[tumor_binary_part == '1'] = '1'
    #convert no to 0
    tumor_binary_part[tumor_binary_part == '2'] = '0'
    # np.where(tumor_binary_part == '?')
    # (array([9, 16]), array([7, 10])) ->> entry [9,7] and [16,10] = '?'
    tumor_binary_part[tumor_binary_part == '?'] = 'nan'


    tumor_binary_part = tumor_binary_part.astype(float)

    # shape (339, 24)
    tumor = np.concatenate((tumor_non_binary_part,tumor_binary_part), axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(tumor == 0, axis=1)
    tumor = tumor[~is_zero_row, :]
    is_zero_col = np.all(tumor == 0, axis=0)
    # shape (339, 24)
    tumor = tumor[:, ~is_zero_col]

    # the final number of missing entries: 2*1 + 67*3 + 155*3 + 2*1
    # the final dimension: 339 x 24
    np.savetxt('./data/tumor_w_missing.txt', tumor, fmt='%s')

def hepatitis():
    # Hepatitis 155 x 38

    # original data of shape (155, 20) with classification column and some non-binary variables
    hep_categorical_w_id_and_classification = np.loadtxt('./data/raw_data/Hepatitis/hepatitis.data.txt', delimiter=',', dtype=object)

    # the first column is the classification attribute
    # we drop this column
    hep_categorical = hep_categorical_w_id_and_classification[:, 1:]

    # 2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
    # binarise by threshold 40
    hep_age = np.zeros([155, 2], dtype=int)
    hep_age[:, 0][hep_categorical[:, 0].astype(int) <= 40] = 1
    hep_age[:, 1][hep_categorical[:, 0].astype(int) > 40] = 1

    # 15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
    #binarise by threshold 2.00
    tmp = np.copy(hep_categorical[:, 13])
    tmp[tmp == '?'] = -1
    tmp = tmp.astype(float)
    hep_bili = np.zeros([155, 2], dtype=int)
    hep_bili[:, 0][tmp <= 2.0] = 1
    hep_bili[:, 1][tmp > 2.0] = 1
    hep_bili[:, 0][hep_categorical[:, 13] == '?'] = 0
    hep_bili[:, 1][hep_categorical[:, 13] == '?'] = 0

    # 16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
    # binarise by threshold 100
    tmp = np.copy(hep_categorical[:, 14])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_alk = np.zeros([155, 2], dtype=int)
    hep_alk[:, 0][tmp <= 100] = 1
    hep_alk[:, 1][tmp > 100] = 1
    hep_alk[:, 0][hep_categorical[:, 14] == '?'] = 0
    hep_alk[:, 1][hep_categorical[:, 14] == '?'] = 0

    # 17. SGOT: 13, 100, 200, 300, 400, 500,
    # binarise by threshold 60
    tmp = np.copy(hep_categorical[:, 15])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_sgot = np.zeros([155, 2], dtype=int)
    hep_sgot[:, 0][tmp <= 60] = 1
    hep_sgot[:, 1][tmp > 60] = 1
    hep_sgot[:, 0][hep_categorical[:, 15] == '?'] = 0
    hep_sgot[:, 1][hep_categorical[:, 15] == '?'] = 0

    # 18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
    # binarise by threshold 4
    tmp = np.copy(hep_categorical[:, 16])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(float)
    hep_alb = np.zeros([155, 2], dtype=int)
    hep_alb[:, 0][tmp <= 4] = 1
    hep_alb[:, 1][tmp > 4] = 1
    hep_alb[:, 0][hep_categorical[:, 16] == '?'] = 0
    hep_alb[:, 1][hep_categorical[:, 16] == '?'] = 0

    # 19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
    # binarise by threshold 65
    tmp = np.copy(hep_categorical[:, 17])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_pro = np.zeros([155, 2], dtype=int)
    hep_pro[:, 0][tmp <= 65] = 1
    hep_pro[:, 1][tmp > 65] = 1
    hep_pro[:, 0][hep_categorical[:, 17] == '?'] = 0
    hep_pro[:, 1][hep_categorical[:, 17] == '?'] = 0

    # 13 columns taking binary values, and has missing values
    # we expand them into 26 Boolean columns
    # 3. SEX : male, female represented as 1, 2
    #  4. STEROID: no, yes
    #  5. ANTIVIRALS: no, yes
    #  6. FATIGUE: no, yes
    #  7. MALAISE: no, yes
    #  8. ANOREXIA: no, yes
    #  9. LIVER BIG: no, yes
    # 10. LIVER FIRM: no, yes
    # 11. SPLEEN PALPABLE: no, yes
    # 12. SPIDERS: no, yes
    # 13. ASCITES: no, yes
    # 14. VARICES: no, yes
    hep_rest = np.zeros([155, 26], dtype=int)
    t = 0
    for i in range(1,13):
        hep_rest[:, t][hep_categorical[:, i] == '1'] = 1
        t += 1
        hep_rest[:, t][hep_categorical[:, i] == '2'] = 1
        t += 1
    # 20. HISTOLOGY: no, yes
    hep_rest[:, 24][hep_categorical[:, 18] == '1'] = 1
    hep_rest[:, 25][hep_categorical[:, 18] == '2'] = 1

    hep = np.concatenate((hep_age, hep_bili,
                                  hep_alk, hep_sgot,
                                  hep_alb, hep_pro, hep_rest), axis=1)


    # binary matrix of shape 155 x 38

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(hep == 0, axis=1)
    hep = hep[~is_zero_row, :]
    is_zero_col = np.all(hep == 0, axis=0)
    hep = hep[:, ~is_zero_col]

    np.savetxt('./data/hepatitis.txt', hep, fmt='%u')

def hepatitis_w_missing():
    # Hepatitis 155 x 38

    # original data of shape (155, 20) with classification column and some non-binary variables
    hep_categorical_w_id_and_classification = np.loadtxt('./data/raw_data/Hepatitis/hepatitis.data.txt', delimiter=',', dtype=object)

    # the first column is the classification attribute
    # we drop this column
    hep_categorical = hep_categorical_w_id_and_classification[:, 1:]

    # 2. AGE: 10, 20, 30, 40, 50, 60, 70, 80        - 0 missing value
    # binarise by threshold 40
    hep_age = np.zeros([155, 2], dtype=int)
    hep_age[:, 0][hep_categorical[:, 0].astype(int) <= 40] = 1
    hep_age[:, 1][hep_categorical[:, 0].astype(int) > 40] = 1

    # 15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
    #binarise by threshold 2.00                     - 6 missing value
    tmp = np.copy(hep_categorical[:, 13])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(float)
    hep_bili = np.zeros([155, 2], dtype=float)
    hep_bili[:, 0][tmp <= 2.0] = 1
    hep_bili[:, 1][tmp > 2.0] = 1
    hep_bili[:, 0][hep_categorical[:, 13] == '?'] = np.nan
    hep_bili[:, 1][hep_categorical[:, 13] == '?'] = np.nan

    # 16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
    # binarise by threshold 100                     - 26 missing value
    tmp = np.copy(hep_categorical[:, 14])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_alk = np.zeros([155, 2], dtype=float)
    hep_alk[:, 0][tmp <= 100] = 1
    hep_alk[:, 1][tmp > 100] = 1
    hep_alk[:, 0][hep_categorical[:, 14] == '?'] = np.nan
    hep_alk[:, 1][hep_categorical[:, 14] == '?'] = np.nan

    # 17. SGOT: 13, 100, 200, 300, 400, 500,
    # binarise by threshold 60                      - 4 missing value
    tmp = np.copy(hep_categorical[:, 15])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_sgot = np.zeros([155, 2], dtype=float)
    hep_sgot[:, 0][tmp <= 60] = 1
    hep_sgot[:, 1][tmp > 60] = 1
    hep_sgot[:, 0][hep_categorical[:, 15] == '?'] = np.nan
    hep_sgot[:, 1][hep_categorical[:, 15] == '?'] = np.nan

    # 18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
    # binarise by threshold 4                       - 16 missing value
    tmp = np.copy(hep_categorical[:, 16])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(float)
    hep_alb = np.zeros([155, 2], dtype=float)
    hep_alb[:, 0][tmp <= 4] = 1
    hep_alb[:, 1][tmp > 4] = 1
    hep_alb[:, 0][hep_categorical[:, 16] == '?'] = np.nan
    hep_alb[:, 1][hep_categorical[:, 16] == '?'] = np.nan

    # 19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
    # binarise by threshold 65                      - 67 missing value
    tmp = np.copy(hep_categorical[:, 17])
    tmp[tmp == '?'] = '-1'
    tmp = tmp.astype(int)
    hep_pro = np.zeros([155, 2], dtype=float)
    hep_pro[:, 0][tmp <= 65] = 1
    hep_pro[:, 1][tmp > 65] = 1
    hep_pro[:, 0][hep_categorical[:, 17] == '?'] = np.nan
    hep_pro[:, 1][hep_categorical[:, 17] == '?'] = np.nan

    # 13 columns taking binary values, and have missing values
    # we expand them into 26 Boolean columns
    # 3. SEX : male, female represented as 1, 2
    #  4. STEROID: no, yes          - 1 missing value
    #  5. ANTIVIRALS: no, yes       - 0 missing value
    #  6. FATIGUE: no, yes          - 1 missing value
    #  7. MALAISE: no, yes          - 1 missing value
    #  8. ANOREXIA: no, yes         - 1 missing value
    #  9. LIVER BIG: no, yes        - 10 missing value
    # 10. LIVER FIRM: no, yes       - 11 missing value
    # 11. SPLEEN PALPABLE: no, yes  - 5 missing value
    # 12. SPIDERS: no, yes          - 5 missing value
    # 13. ASCITES: no, yes          - 5 missing value
    # 14. VARICES: no, yes          - 5 missing value
    hep_rest = np.zeros([155, 26], dtype=float)
    t = 0
    for i in range(1,13):
        hep_rest[:, t][hep_categorical[:, i] == '1'] = 1
        hep_rest[:, t][hep_categorical[:, i] == '?'] = np.nan
        t += 1
        hep_rest[:, t][hep_categorical[:, i] == '2'] = 1
        hep_rest[:, t][hep_categorical[:, i] == '?'] = np.nan
        t += 1
    # 20. HISTOLOGY: no, yes         - 0 missing value
    hep_rest[:, 24][hep_categorical[:, 18] == '1'] = 1
    hep_rest[:, 25][hep_categorical[:, 18] == '2'] = 1

    hep = np.concatenate((hep_age, hep_bili,
                                  hep_alk, hep_sgot,
                                  hep_alb, hep_pro, hep_rest), axis=1)


    # binary matrix of shape 155 x 38

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(hep == 0, axis=1)
    hep = hep[~is_zero_row, :]
    is_zero_col = np.all(hep == 0, axis=0)
    hep = hep[:, ~is_zero_col]

    np.savetxt('./data/hepatitis_w_missing.txt', hep, fmt='%s')

def heart():
    #SPECT HEART DATA, 242 x 22

    #no missing values

    #dataset is divided into two parts
    heart_test = np.loadtxt('./data/raw_data/SPECT_Heart/SPECT.test.txt', delimiter=',', dtype=int)
    heart_train = np.loadtxt('./data/raw_data/SPECT_Heart/SPECT.train.txt', delimiter=',', dtype=int)
    # (267, 23)
    heart_w_classification_column = np.concatenate((heart_test,heart_train))
    #the first column corresponds to classifications of OVERALL_DIAGNOSIS, we delete this column
    # shape (267, 22)
    heart = heart_w_classification_column[:,1:]

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(heart == 0, axis=1)
    heart = heart[~is_zero_row, :]
    is_zero_col = np.all(heart == 0, axis=0)
    # shape (242, 22)
    heart = heart[:, ~is_zero_col]

    np.savetxt('./data/heart.txt', heart, fmt='%u')

def lymph():
    # Lymphography (148, 44)

    # no missing values

    # original data of shape (148, 19) with classification column and some non-binary variables
    lymp_cat = np.loadtxt('./data/raw_data/Lymphography/lymphography.data.txt', delimiter=',',
                                                         dtype=object)

    # 2. lymphatics: normal, arched, deformed, displaced
    lymp_2 = np.zeros([148, 4], dtype=int)
    lymp_2[:, 0][lymp_cat[:, 1] == '1'] = 1
    lymp_2[:, 1][lymp_cat[:, 1] == '2'] = 1
    lymp_2[:, 2][lymp_cat[:, 1] == '3'] = 1
    lymp_2[:, 3][lymp_cat[:, 1] == '4'] = 1

    # 10. lym.nodes dimin: 0-3
    lymp_10 = np.zeros([148, 3], dtype=int)
    lymp_10[:, 0][lymp_cat[:, 9] == '1'] = 1
    lymp_10[:, 1][lymp_cat[:, 9] == '2'] = 1
    lymp_10[:, 2][lymp_cat[:, 9] == '3'] = 1

    # 11. lym.nodes enlar: 1-4
    lymp_11 = np.zeros([148, 4], dtype=int)
    lymp_11[:, 0][lymp_cat[:, 10] == '1'] = 1
    lymp_11[:, 1][lymp_cat[:, 10] == '2'] = 1
    lymp_11[:, 2][lymp_cat[:, 10] == '3'] = 1
    lymp_11[:, 3][lymp_cat[:, 10] == '4'] = 1

    # 12. changes in lym.: bean, oval, round
    lymp_12 = np.zeros([148, 3], dtype=int)
    lymp_12[:, 0][lymp_cat[:, 11] == '1'] = 1
    lymp_12[:, 1][lymp_cat[:, 11] == '2'] = 1
    lymp_12[:, 2][lymp_cat[:, 11] == '3'] = 1

    # 13. defect in node: no, lacunar, lac. marginal, lac. central
    lymp_13 = np.zeros([148, 4], dtype=int)
    lymp_13[:, 0][lymp_cat[:, 12] == '1'] = 1
    lymp_13[:, 1][lymp_cat[:, 12] == '2'] = 1
    lymp_13[:, 2][lymp_cat[:, 12] == '3'] = 1
    lymp_13[:, 3][lymp_cat[:, 12] == '4'] = 1

    # 14. changes in node: no, lacunar, lac. margin, lac. central
    lymp_14 = np.zeros([148, 4], dtype=int)
    lymp_14[:, 0][lymp_cat[:, 13] == '1'] = 1
    lymp_14[:, 1][lymp_cat[:, 13] == '2'] = 1
    lymp_14[:, 2][lymp_cat[:, 13] == '3'] = 1
    lymp_14[:, 3][lymp_cat[:, 13] == '4'] = 1

    # 15. changes in stru: no, grainy, drop-like, coarse, diluted, reticular,
    #                         stripped, faint,
    lymp_15 = np.zeros([148, 8], dtype=int)
    lymp_15[:, 0][lymp_cat[:, 14] == '1'] = 1
    lymp_15[:, 1][lymp_cat[:, 14] == '2'] = 1
    lymp_15[:, 2][lymp_cat[:, 14] == '3'] = 1
    lymp_15[:, 3][lymp_cat[:, 14] == '4'] = 1
    lymp_15[:, 4][lymp_cat[:, 14] == '5'] = 1
    lymp_15[:, 5][lymp_cat[:, 14] == '6'] = 1
    lymp_15[:, 6][lymp_cat[:, 14] == '7'] = 1
    lymp_15[:, 7][lymp_cat[:, 14] == '8'] = 1

    # 16. special forms: no, chalices, vesicles
    lymp_16 = np.zeros([148, 3], dtype=int)
    lymp_16[:, 0][lymp_cat[:, 15] == '1'] = 1
    lymp_16[:, 1][lymp_cat[:, 15] == '2'] = 1
    lymp_16[:, 2][lymp_cat[:, 15] == '3'] = 1

    # 19. no. of nodes in: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70
    # binarise at threshold 40
    lymp_19 = np.zeros([148, 2], dtype=int)
    lymp_19[:, 0][lymp_cat[:, 18].astype(int) <= 4] = 1
    lymp_19[:, 1][lymp_cat[:, 18].astype(int) > 4]= 1


    # 3. block of affere: no, yes
    # 4. bl. of lymph. c: no, yes
    # 5. bl. of lymph. s: no, yes
    # 6. by pass: no, yes
    # 7. extravasates: no, yes
    # 8. regeneration of: no, yes
    # 9. early uptake in: no, yes
    # 17. dislocation of: no, yes
    # 18. exclusion of no: no, yes

    lymp_bool = np.zeros([148, 9], dtype=int)
    cols = [2,3,4,5,6,7,8,16,17]
    for i in range(9):
        lymp_bool[:, i][lymp_cat[:, cols[i]] == '2'] = 1

    lymp = np.concatenate((lymp_2, lymp_10, lymp_11, lymp_12, lymp_13,
                           lymp_14, lymp_15, lymp_16, lymp_19, lymp_bool), axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(lymp == 0, axis=1)
    lymp = lymp[~is_zero_row, :]
    is_zero_col = np.all(lymp == 0, axis=0)
    lymp = lymp[:, ~is_zero_col]

    np.savetxt('./data/lymp.txt', lymp, fmt='%u')

def votes():
    #VOTES 434 x 32

    #original data of shape (435, 17) with classification column
    votes_categorical_w_classification = np.loadtxt('./data/raw_data/Votes/house-votes-84.data.txt', delimiter=',', dtype = object)
    # shape (435, 16)
    votes_categorical = votes_categorical_w_classification[:,1:]
    votes = np.zeros([435,32],dtype=int)

    k = 0
    for i in range(16):
        votes[:, k][votes_categorical[:, i] == 'y'] = 1
        k += 1
        votes[:, k][votes_categorical[:, i] == 'n'] = 1
        k += 1


    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(votes == 0, axis=1)
    votes = votes[~is_zero_row, :]
    is_zero_col = np.all(votes == 0, axis=0)
    votes = votes[:, ~is_zero_col]

    np.savetxt('./data/votes.txt', votes, fmt='%u')

def votes_w_missing():
    #VOTES 435 x 16

    #original data of shape (435, 17) with classification column
    votes_categorical_w_classification = np.loadtxt('./data/raw_data/Votes/house-votes-84.data.txt', delimiter=',', dtype = object)
    # shape (435, 16)
    votes_categorical = votes_categorical_w_classification[:,1:]
    votes = np.zeros([435,16],dtype=float)

    for i in range(16):
        votes[:, i][votes_categorical[:, i] == 'y'] = 1
        votes[:, i][votes_categorical[:, i] == 'n'] = 0
        votes[:, i][votes_categorical[:, i] == '?'] = np.nan

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(votes == 0, axis=1)
    votes = votes[~is_zero_row, :]
    is_zero_col = np.all(votes == 0, axis=0)
    votes = votes[:, ~is_zero_col]

    np.savetxt('./data/votes_w_missing.txt', votes, fmt='%s')

def votes_w_missing2():
    #VOTES 435 x 32

    #original data of shape (435, 17) with classification column
    votes_categorical_w_classification = np.loadtxt('./data/raw_data/Votes/house-votes-84.data.txt', delimiter=',', dtype = object)
    # shape (435, 16)
    votes_categorical = votes_categorical_w_classification[:,1:]
    votes = np.zeros([435,32],dtype=float)

    k = 0
    for i in range(16):
        votes[:, k][votes_categorical[:, i] == 'y'] = 1
        votes[:, k][votes_categorical[:, i] == '?'] = np.nan
        k += 1
        votes[:, k][votes_categorical[:, i] == 'n'] = 1
        votes[:, k][votes_categorical[:, i] == '?'] = np.nan
        k += 1

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(votes == 0, axis=1)
    votes = votes[~is_zero_row, :]
    is_zero_col = np.all(votes == 0, axis=0)
    votes = votes[:, ~is_zero_col]

    np.savetxt('./data/votes_w_missing2.txt', votes, fmt='%s')

def audio():

    # Audiology Standarised, 226 x 69

    # Number of instances: 200 training cases, 26 test cases
    # Number of attributes: 69 + identifier attribute + class attribute
    audio_train = np.loadtxt('./data/raw_data/Audiology/audiology.standardized.data.txt', delimiter=',', dtype=object)
    audio_test = np.loadtxt('./data/raw_data/Audiology/audiology.standardized.test.txt', delimiter=',', dtype=object)

    # shape (226, 71)
    audio_w_id_class_cols = np.concatenate((audio_test,audio_train))

    # the penultimate column is an identifier unique for each instance
    # the last column is the classification class
    # we delete these columns
    # shape (226, 69)
    audio_categorical = audio_w_id_class_cols[:,:-2]

    # audio_categorical is 226 x 69 with 9 categorical columns, the rest is Boolean


    # col 1, air(): mild,moderate,severe,normal,profound
    air_mild = np.zeros([226, 1],dtype=int)
    air_mild[audio_categorical[:, 1] == 'mild'] = 1
    air_mod = np.zeros([226, 1], dtype=int)
    air_mod[audio_categorical[:, 1] == 'moderate'] = 1
    air_sev = np.zeros([226, 1], dtype=int)
    air_sev[audio_categorical[:, 1] == 'severe'] = 1
    air_norm = np.zeros([226, 1], dtype=int)
    air_norm[audio_categorical[:, 1] == 'normal'] = 1
    air_pro = np.zeros([226, 1], dtype=int)
    air_pro[audio_categorical[:, 1] == 'profound'] = 1
    air = np.concatenate((air_mild, air_mod, air_sev, air_norm, air_pro),axis=1)

    # col 3, ar_c(): normal,elevated,absent
    ar_c_norm = np.zeros([226, 1],dtype=int)
    ar_c_norm[audio_categorical[:, 3] == 'normal'] = 1
    ar_c_ele = np.zeros([226, 1],dtype=int)
    ar_c_ele[audio_categorical[:, 3] == 'elevated'] = 1
    ar_c_abs = np.zeros([226, 1],dtype=int)
    ar_c_abs[audio_categorical[:, 3] == 'absent'] = 1
    ar_c = np.concatenate((ar_c_norm, ar_c_ele, ar_c_abs), axis=1)

    # col 4, ar_u(): normal,elevated,absent
    ar_u_norm = np.zeros([226, 1],dtype=int)
    ar_u_norm[audio_categorical[:, 4] == 'normal'] = 1
    ar_u_ele = np.zeros([226, 1],dtype=int)
    ar_u_ele[audio_categorical[:, 4] == 'elevated'] = 1
    ar_u_abs = np.zeros([226, 1],dtype=int)
    ar_u_abs[audio_categorical[:, 4] == 'absent'] = 1
    ar_u = np.concatenate((ar_u_norm, ar_u_ele, ar_u_abs), axis=1)

    # col 5, bone(): mild,moderate,normal,unmeasured
    bone_mild = np.zeros([226, 1],dtype=int)
    bone_mild[audio_categorical[:, 5] == 'mild'] = 1
    bone_mod = np.zeros([226, 1], dtype=int)
    bone_mod[audio_categorical[:, 5] == 'moderate'] = 1
    bone_norm = np.zeros([226, 1], dtype=int)
    bone_norm[audio_categorical[:, 5] == 'normal'] = 1
    bone_un = np.zeros([226, 1], dtype=int)
    bone_un[audio_categorical[:, 5] == 'unmeasured'] = 1
    bone = np.concatenate((bone_mild, bone_mod, bone_norm,bone_un),axis=1)

    # col 7, bser(): normal,degraded
    bser_norm = np.zeros([226, 1], dtype=int)
    bser_norm[audio_categorical[:, 7] == 'normal'] = 1
    bser_deg = np.zeros([226, 1], dtype=int)
    bser_deg[audio_categorical[:, 7] == 'degraded'] = 1
    bser = np.concatenate((bser_norm,bser_deg),axis=1)

    # col 58, o_ar_c():	normal,elevated,absent
    o_ar_c_norm = np.zeros([226, 1], dtype=int)
    o_ar_c_norm[audio_categorical[:, 58] == 'normal'] = 1
    o_ar_c_ele = np.zeros([226, 1], dtype=int)
    o_ar_c_ele[audio_categorical[:, 58] == 'elevated'] = 1
    o_ar_c_abs = np.zeros([226, 1], dtype=int)
    o_ar_c_abs[audio_categorical[:, 58] == 'absent'] = 1
    o_ar_c = np.concatenate((o_ar_c_norm,o_ar_c_ele,o_ar_c_abs),axis=1)

    # col 59, o_ar_u():	normal,elevated,absent
    o_ar_u_norm = np.zeros([226, 1], dtype=int)
    o_ar_u_norm[audio_categorical[:, 59] == 'normal'] = 1
    o_ar_u_ele = np.zeros([226, 1], dtype=int)
    o_ar_u_ele[audio_categorical[:, 59] == 'elevated'] = 1
    o_ar_u_abs = np.zeros([226, 1], dtype=int)
    o_ar_u_abs[audio_categorical[:, 59] == 'absent'] = 1
    o_ar_u = np.concatenate((o_ar_u_norm,o_ar_u_ele,o_ar_u_abs),axis=1)

    # col 63, speech(): normal,good,very_good,very_poor,poor,unmeasured
    speech_norm = np.zeros([226, 1], dtype=int)
    speech_norm[audio_categorical[:, 63] == 'normal'] = 1
    speech_good = np.zeros([226, 1], dtype=int)
    speech_good[audio_categorical[:, 63] == 'good'] = 1
    speech_vgood = np.zeros([226, 1], dtype=int)
    speech_vgood[audio_categorical[:, 63] == 'very_good'] = 1
    speech_vpoor = np.zeros([226, 1], dtype=int)
    speech_vpoor[audio_categorical[:, 63] == 'very_poor'] = 1
    speech_poor = np.zeros([226, 1], dtype=int)
    speech_poor[audio_categorical[:, 63] == 'poor'] = 1
    speech_un = np.zeros([226, 1], dtype=int)
    speech_un[audio_categorical[:, 63] == 'unmeasured'] = 1
    speech = np.concatenate((speech_norm, speech_good, speech_vgood, speech_vpoor, speech_poor, speech_un), axis=1)

    # col 65, tymp(): a,as,b,ad,c
    tymp_a = np.zeros([226, 1], dtype=int)
    tymp_a[audio_categorical[:, 65] == 'a'] = 1
    tymp_as = np.zeros([226, 1], dtype=int)
    tymp_as[audio_categorical[:, 65] == 'as'] = 1
    tymp_b = np.zeros([226, 1], dtype=int)
    tymp_b[audio_categorical[:, 65] == 'b'] = 1
    tymp_ad = np.zeros([226, 1], dtype=int)
    tymp_ad[audio_categorical[:, 65] == 'ad'] = 1
    tymp_c = np.zeros([226, 1], dtype=int)
    tymp_c[audio_categorical[:, 65] == 'c'] = 1
    tymp = np.concatenate((tymp_a,tymp_as,tymp_b,tymp_ad,tymp_c),axis=1)

    # convert the Boolean columns to numerical value
    idx_Boolean = np.arange(0,69)
    idx_Boolean = np.delete(idx_Boolean, [1, 3, 4, 5, 7, 58, 59, 63, 65])
    audio_Boolean = audio_categorical[:,idx_Boolean]

    audio_Boolean[audio_Boolean == 't'] = 1
    audio_Boolean[audio_Boolean == 'f'] = 0

    # can check that there were no missing values in the Boolean column by:
    #np.unique(audio_Boolean)

    audio_Boolean = audio_Boolean.astype(int)

    # shape (226, 94)
    audio = np.concatenate((audio_Boolean, air, ar_c, ar_u, bone, bser, o_ar_c, o_ar_u, speech, tymp), axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(audio == 0, axis=1)
    audio = audio[~is_zero_row, :]
    is_zero_col = np.all(audio == 0, axis=0)
    # shape (226, 94)
    audio = audio[:, ~is_zero_col]


    np.savetxt('./data/audio.txt', audio, fmt='%u')

def audio_w_missing():

    # Audiology Standarised, 226 x 92

    # Number of instances: 200 training cases, 26 test cases
    # Number of attributes: 69 + identifier attribute + class attribute
    audio_train = np.loadtxt('./data/raw_data/Audiology/audiology.standardized.data.txt', delimiter=',', dtype=object)
    audio_test = np.loadtxt('./data/raw_data/Audiology/audiology.standardized.test.txt', delimiter=',', dtype=object)

    # shape (226, 71)
    audio_w_id_class_cols = np.concatenate((audio_test,audio_train))

    # the penultimate column is an identifier unique for each instance
    # the last column is the classification class
    # we delete these columns
    # shape (226, 69)
    audio_categorical = audio_w_id_class_cols[:,:-2]

    # audio_categorical is 226 x 69 with 9 categorical columns, the rest is Boolean


    # col 1, air(): mild,moderate,severe,normal,profound        0 missing values
    air_mild = np.zeros([226, 1],dtype=int)
    air_mild[audio_categorical[:, 1] == 'mild'] = 1
    air_mod = np.zeros([226, 1], dtype=int)
    air_mod[audio_categorical[:, 1] == 'moderate'] = 1
    air_sev = np.zeros([226, 1], dtype=int)
    air_sev[audio_categorical[:, 1] == 'severe'] = 1
    air_norm = np.zeros([226, 1], dtype=int)
    air_norm[audio_categorical[:, 1] == 'normal'] = 1
    air_pro = np.zeros([226, 1], dtype=int)
    air_pro[audio_categorical[:, 1] == 'profound'] = 1
    air = np.concatenate((air_mild, air_mod, air_sev, air_norm, air_pro),axis=1)

    # col 3, ar_c(): normal,elevated,absent                 4 missing values
    ar_c_norm = np.zeros([226, 1])
    ar_c_norm[audio_categorical[:, 3] == 'normal'] = 1
    ar_c_norm[audio_categorical[:, 3] == '?'] = np.nan
    ar_c_ele = np.zeros([226, 1])
    ar_c_ele[audio_categorical[:, 3] == 'elevated'] = 1
    ar_c_ele[audio_categorical[:, 3] == '?'] = np.nan
    ar_c_abs = np.zeros([226, 1])
    ar_c_abs[audio_categorical[:, 3] == 'absent'] = 1
    ar_c_abs[audio_categorical[:, 3] == '?'] = np.nan
    ar_c = np.concatenate((ar_c_norm, ar_c_ele, ar_c_abs), axis=1)

    # col 4, ar_u(): normal,elevated,absent                 3 missing values
    ar_u_norm = np.zeros([226, 1])
    ar_u_norm[audio_categorical[:, 4] == 'normal'] = 1
    ar_u_norm[audio_categorical[:, 4] == '?'] = np.nan
    ar_u_ele = np.zeros([226, 1])
    ar_u_ele[audio_categorical[:, 4] == 'elevated'] = 1
    ar_u_ele[audio_categorical[:, 4] == '?'] = np.nan
    ar_u_abs = np.zeros([226, 1])
    ar_u_abs[audio_categorical[:, 4] == 'absent'] = 1
    ar_u_abs[audio_categorical[:, 4] == '?'] = np.nan
    ar_u = np.concatenate((ar_u_norm, ar_u_ele, ar_u_abs), axis=1)

    # col 5, bone(): mild,moderate,normal,unmeasured           75 missing values + 46 unmeasured entries = all treated as missing values
    bone_mild = np.zeros([226, 1])
    bone_mild[audio_categorical[:, 5] == 'mild'] = 1
    bone_mild[audio_categorical[:, 5] == 'unmeasured'] = np.nan
    bone_mild[audio_categorical[:, 5] == '?'] = np.nan
    bone_mod = np.zeros([226, 1])
    bone_mod[audio_categorical[:, 5] == 'moderate'] = 1
    bone_mod[audio_categorical[:, 5] == 'unmeasured'] = np.nan
    bone_mod[audio_categorical[:, 5] == '?'] = np.nan
    bone_norm = np.zeros([226, 1])
    bone_norm[audio_categorical[:, 5] == 'normal'] = 1
    bone_norm[audio_categorical[:, 5] == 'unmeasured'] = np.nan
    bone_norm[audio_categorical[:, 5] == '?'] = np.nan
    bone = np.concatenate((bone_mild, bone_mod, bone_norm),axis=1)

    # col 7, bser(): normal,degraded                    222 missing values
    bser_norm = np.zeros([226, 1])
    bser_norm[audio_categorical[:, 7] == 'normal'] = 1
    bser_norm[audio_categorical[:, 7] == '?'] = np.nan
    bser_deg = np.zeros([226, 1])
    bser_deg[audio_categorical[:, 7] == 'degraded'] = 1
    bser_deg[audio_categorical[:, 7] == '?'] = np.nan
    bser = np.concatenate((bser_norm,bser_deg),axis=1)

    # col 58, o_ar_c():	normal,elevated,absent          5 missing values
    o_ar_c_norm = np.zeros([226, 1])
    o_ar_c_norm[audio_categorical[:, 58] == 'normal'] = 1
    o_ar_c_norm[audio_categorical[:, 58] == '?'] = np.nan
    o_ar_c_ele = np.zeros([226, 1])
    o_ar_c_ele[audio_categorical[:, 58] == 'elevated'] = 1
    o_ar_c_ele[audio_categorical[:, 58] == '?'] = np.nan
    o_ar_c_abs = np.zeros([226, 1])
    o_ar_c_abs[audio_categorical[:, 58] == 'absent'] = 1
    o_ar_c_abs[audio_categorical[:, 58] == '?'] = np.nan
    o_ar_c = np.concatenate((o_ar_c_norm,o_ar_c_ele,o_ar_c_abs),axis=1)

    # col 59, o_ar_u():	normal,elevated,absent          2 missing values
    o_ar_u_norm = np.zeros([226, 1])
    o_ar_u_norm[audio_categorical[:, 59] == 'normal'] = 1
    o_ar_u_norm[audio_categorical[:, 59] == '?'] = np.nan
    o_ar_u_ele = np.zeros([226, 1])
    o_ar_u_ele[audio_categorical[:, 59] == 'elevated'] = 1
    o_ar_u_ele[audio_categorical[:, 59] == '?'] = np.nan
    o_ar_u_abs = np.zeros([226, 1])
    o_ar_u_abs[audio_categorical[:, 59] == 'absent'] = 1
    o_ar_u_abs[audio_categorical[:, 59] == '?'] = np.nan
    o_ar_u = np.concatenate((o_ar_u_norm,o_ar_u_ele,o_ar_u_abs),axis=1)

    # col 63, speech(): normal,good,very_good,very_poor,poor,unmeasured     6 missing values + 4 unmeasured treated as missing valeus
    speech_norm = np.zeros([226, 1])
    speech_norm[audio_categorical[:, 63] == 'normal'] = 1
    speech_norm[audio_categorical[:, 63] == '?'] = np.nan
    speech_norm[audio_categorical[:, 63] == 'unmeasured'] = np.nan
    speech_good = np.zeros([226, 1])
    speech_good[audio_categorical[:, 63] == 'good'] = 1
    speech_good[audio_categorical[:, 63] == '?'] = np.nan
    speech_good[audio_categorical[:, 63] == 'unmeasured'] = np.nan
    speech_vgood = np.zeros([226, 1])
    speech_vgood[audio_categorical[:, 63] == 'very_good'] = 1
    speech_vgood[audio_categorical[:, 63] == '?'] = np.nan
    speech_vgood[audio_categorical[:, 63] == 'unmeasured'] = np.nan
    speech_vpoor = np.zeros([226, 1])
    speech_vpoor[audio_categorical[:, 63] == 'very_poor'] = 1
    speech_vpoor[audio_categorical[:, 63] == '?'] = np.nan
    speech_vpoor[audio_categorical[:, 63] == 'unmeasured'] = np.nan
    speech_poor = np.zeros([226, 1])
    speech_poor[audio_categorical[:, 63] == 'poor'] = 1
    speech_poor[audio_categorical[:, 63] == '?'] = np.nan
    speech_poor[audio_categorical[:, 63] == 'unmeasured'] = np.nan
    speech = np.concatenate((speech_norm, speech_good, speech_vgood, speech_vpoor, speech_poor), axis=1)

    # col 65, tymp(): a,as,b,ad,c           0 missing values
    tymp_a = np.zeros([226, 1], dtype=int)
    tymp_a[audio_categorical[:, 65] == 'a'] = 1
    tymp_as = np.zeros([226, 1], dtype=int)
    tymp_as[audio_categorical[:, 65] == 'as'] = 1
    tymp_b = np.zeros([226, 1], dtype=int)
    tymp_b[audio_categorical[:, 65] == 'b'] = 1
    tymp_ad = np.zeros([226, 1], dtype=int)
    tymp_ad[audio_categorical[:, 65] == 'ad'] = 1
    tymp_c = np.zeros([226, 1], dtype=int)
    tymp_c[audio_categorical[:, 65] == 'c'] = 1
    tymp = np.concatenate((tymp_a,tymp_as,tymp_b,tymp_ad,tymp_c),axis=1)

    # convert the Boolean columns to numerical value     0 missing values
    idx_Boolean = np.arange(0,69)
    idx_Boolean = np.delete(idx_Boolean, [1, 3, 4, 5, 7, 58, 59, 63, 65])
    audio_Boolean = audio_categorical[:,idx_Boolean]

    audio_Boolean[audio_Boolean == 't'] = 1
    audio_Boolean[audio_Boolean == 'f'] = 0

    # can check that there were no missing values in the Boolean column by:
    #np.unique(audio_Boolean)

    audio_Boolean = audio_Boolean.astype(int)

    # shape (226, 92)
    audio = np.concatenate((audio_Boolean, air, ar_c, ar_u, bone, bser, o_ar_c, o_ar_u, speech, tymp), axis=1)

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(audio == 0, axis=1)
    audio = audio[~is_zero_row, :]
    is_zero_col = np.all(audio == 0, axis=0)
    # shape (226, 94)
    audio = audio[:, ~is_zero_col]


    np.savetxt('./data/audio_w_missing.txt', audio, fmt='%s')

def apb():

    import networkx as nx

    # no missing values

    # APB 105 x 105

    apb_graph = nx.read_edgelist('./data/raw_data/APB/books.dat.txt')

    apb = nx.adjacency_matrix(apb_graph).todense()

    apb = np.asarray(apb)

    # apb = nx.incidence_matrix(apb_graph).todense()
    #
    # apb = np.asarray(apb).transpose()

    # check if there is a row or column which is only 0s,
    # that does not carry any information for factorisation and can be reinserted at any time
    is_zero_row = np.all(apb == 0, axis=1)
    apb = apb[~is_zero_row, :]
    is_zero_col = np.all(apb == 0, axis=0)
    apb = apb[:, ~is_zero_col]


    np.savetxt('./data/apb.txt', apb, fmt='%u')

if __name__ == "__main__":

    print('\nbinarising data\n')

    zoo()
    print('zoo done')
    heart()
    print('heart done')
    lymph()
    print('lymp done')
    apb()
    print('apb done')

    print('\nbinarising data with missing entries\n')

    tumor_w_missing()
    print('tumor done')
    hepatitis_w_missing()
    print('hepatitis done')
    audio_w_missing()
    print('audio done')
    votes_w_missing()
    print('votes done')

    datasets = ['zoo', 'heart', 'lymp', 'apb', 'tumor_w_missing', 'hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']
    table = pd.DataFrame(columns=datasets, index=['n','m'])

    for name in datasets:
        X = np.loadtxt('./data/' + name + '.txt', dtype=float)
        (n,m) = X.shape
        table.loc['n', name] = n
        table.loc['m', name] = m
        table.loc['#missing', name] = (X!=X).sum()
        table.loc['%missing', name] = (X!=X).sum()/(n*m)
        table.loc['%0s', name] = 100 * (1- np.nansum(X)/(n*m))
        table.loc['%1s', name] = 100 * np.nansum(X) / (n * m)

    table.to_csv('./data/real_data_stats.csv')
    print('\nbinary data saved to ./data\n')
    print(table)


