from glob import glob
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import cv2 as cv
from sklearn.model_selection import train_test_split

class Pipeline(object):

    def __init__(self, list_train, Normalize=True):
        self.scans_train = list_train
        self.train_im = self.read_scans(Normalize)

    def read_scans(self, Normalize):

        train_im = []
        sz = len(self.scans_train)
        # print(sz)
        for i in range(sz):
            if i % 10 == 0:
                print('iteration [{}]'.format(i))

            flair = glob(self.scans_train[i] + '/*_flair.nii.gz')
            t2 = glob(self.scans_train[i] + '/*_t2.nii.gz')
            gt = glob(self.scans_train[i] + '/*_seg.nii.gz')
            t1 = glob(self.scans_train[i] + '/*_t1.nii.gz')
            t1c = glob(self.scans_train[i] + '/*_t1ce.nii.gz')
            if len(flair) <= 0:
                continue
            # print(flair[0])
            scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
            # P_Data = self.Data_Preprocessing(scans)
            # train_im.append(P_Data)
            tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]
            # print(tmp)
            tmp = np.array(tmp)
            # print(tmp.shape)
            # tmp = tmp[:, 1:147, 29:221, 42:194]
            train_im.append(tmp)
            del tmp
        print(np.array(train_im).shape)
        # # np.swapaxes(train_im, 0, 1)
        # print(np.array(train_im).shape)
        return np.array(train_im)
        print('bye')

    def Data_Concatenate(self):
        Input_Data = self.train_im
        Output = []
        for i in range(5):
            print('$')
            c = 0
            counter = 0
            for ii in range(len(Input_Data)):
                if counter < len(Input_Data):
                    a = Input_Data[counter][i, :, :, :]
                    if counter == 0:
                        c = a
                        print('c={}'.format(c.shape))
                    else:
                        c = np.concatenate((a, c), axis=0)
                        print('c={}'.format(c.shape))
                    counter = counter + 1

            # c = c[:, :, :, np.newaxis]
            # print(c.shape)
            Output.append(c)
        return Output

    def Data(self):
        self.InData = self.Data_Concatenate()
        self.InData = np.array(self.InData, dtype='float32')
        print(self.InData.shape)
        x = self.InData[1, :, :, :]
        y = self.InData[4, :, :, :]
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=32)
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        return X_train, X_test, Y_train, Y_test


# if __name__ == '__main__':
path = glob('MICCAI_BraTS2020_TrainingData/**')
start = 0
end = 20
# print(len(path))
pipe = Pipeline(list_train=path[0:20], Normalize=True)
# plt.imshow(pipe.train_im[1][3, :, :, 100])
# plt.show()
X_train, X_test, Y_train, Y_test = pipe.Data()

print('hi')
