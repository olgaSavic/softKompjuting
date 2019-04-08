import os
import sys
import time
import numpy as np
from keras.models import load_model
import cv2
import itertools
import math
from scipy import ndimage
import detekcijaLinijaHough as dh 
from basicFunctions import distance2D, pnt2line
from Obucavanje2 import trainingNeuralNetwork
from NeuronskaMrezaNew import trainNetworkNew, prepare_for_ann, display_result
import slika as sl
from keras.models import model_from_json
import matplotlib.pyplot as plt

# for resize_region
from random import randint


#MODELI 

# model 2
if os.path.exists('model2.h5') is False:
    print('modelNew.h5 fajl ne postoji! Treniram mrezu...')
    trainingNeuralNetwork()

model2 = load_model('model2.h5')
model2.load_weights('model2.h5')

# model New
# ucitavanje podataka iz json fajla 
if os.path.exists('modelNew.h5') is False:
    print('modelNew.h5 fajl ne postoji! Treniram mrezu...')
    trainNetworkNew()

pomFajl = open('modelNew.json', 'r')
mrezaFajl = pomFajl.read()
pomFajl.close()
ann = model_from_json(mrezaFajl)
ann.load_weights("modelNew.h5")

# model New2
if os.path.exists('modelNew2.h5') is False:
    print('modelNew2.h5 fajl ne postoji! Treniram mrezu...')
    trainNetworkNew()

pomFajl2 = open('modelNew2.json', 'r')
mrezaFajl2 = pomFajl2.read()
pomFajl2.close()
ann2 = model_from_json(mrezaFajl2)
ann2.load_weights("modelNew2.h5")


def deskew(img): # za brojeve koji nisu uspravni
    m = cv2.moments(img)

    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11']/m['mu02']
    M = np.array([[1, skew, -0.5*28*skew], [0, 1, 0]], 'float32')
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 222, 255, cv2.THRESH_BINARY)
    return image_bin

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    
    top = int(14 - region.shape[0]/2)  # shape[0] = rows
    bottom = top
    left = int(14 - region.shape[1]/2)  # shape[1] = cols
    right = left

    borderType = cv2.BORDER_CONSTANT
    borderType2 = cv2.BORDER_REPLICATE

    
    if (region.shape[0] == 28 and region.shape[1] == 28):
        print('Dobre dimenzije vol 1.')
    else:     
        value = [255, 255, 255]
        region = cv2.copyMakeBorder(region, top, bottom, left, right, borderType2, None, value)
 
    if (region.shape[0] == 27):
        region = cv2.copyMakeBorder(region, 1, 0, 0, 0, borderType2, None, value)
    if (region.shape[0] == 29):
        region = cv2.copyMakeBorder(region, -1, 0, 0, 0, borderType2, None, value)
    if (region.shape[1] == 27):
        region = cv2.copyMakeBorder(region, 0, 0, 1, 0, borderType2, None, value)
    if (region.shape[1] == 29):
        region = cv2.copyMakeBorder(region, 0, 0, -1, 0, borderType2, None, value)

    #plt.imshow(region, 'gray')
    #plt.show()
        
    return region
    

def select_roi(image_orig, image_bin, plavaLinija, zelenaLinija): # image_orig je frame iz main

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_arrayPlava = []
    regions_arrayZelena = []

    tacke_arrayPlava = []
    tacke_arrayZelena = []

    
    [(xPlavaD, yPlavaG), (xPlavaG, yPlavaD)] = plavaLinija 

    [(xZelenaD, yZelenaG), (xZelenaG, yZelenaD)] = zelenaLinija
    
    # linije
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika

        distancePlava, _ , preciCePlavu = pnt2line((x,y), (xPlavaD, yPlavaG),  (xPlavaG, yPlavaD))
        #print('Distanca od plave linije je: ', distancePlava)

        distanceZelena, _ , preciCeZelenu = pnt2line((x,y), (xZelenaD, yZelenaG),  (xZelenaG, yZelenaD))
        #print('Distanca od zelene linije je: ', distanceZelena)

        area = cv2.contourArea(contour)

        if ((h >= 10 and w >= 6) and (h <= 28 and w <= 28)):
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

            # sabiranje - plava linija
            if (preciCePlavu == True and distancePlava <= 0.5):

                tackaPlava = (x,y)
                tacke_arrayPlava.append(tackaPlava)

                regionPlava = image_bin[y:y+h+1,x:x+w+1]
                regions_arrayPlava.append([resize_region(regionPlava), (x,y,w,h)])       
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(100,80,110),2)

            # oduzimanje - zelena linija
            if (preciCeZelenu == True and distanceZelena <= 0.5):

                tackaZelena = (x,y)
                tacke_arrayZelena.append(tackaZelena)

                regionZelena = image_bin[y:y+h+1,x:x+w+1]
                regions_arrayZelena.append([resize_region(regionZelena), (x,y,w,h)])       
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(100,80,110),2)

    # sortirani regioni za plavu liniju
    regions_arrayPlava = sorted(regions_arrayPlava, key=lambda item: item[1][0])
    sorted_regionsPlava = sorted_regionsPlava = [region[0] for region in regions_arrayPlava]

    # sortirani regioni za zelenu liniju
    regions_arrayZelena = sorted(regions_arrayZelena, key=lambda item: item[1][0])
    sorted_regionsZelena = sorted_regionsZelena = [region[0] for region in regions_arrayZelena]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regionsPlava, sorted_regionsZelena, x, y, tacke_arrayPlava, tacke_arrayZelena


# main funkcija
def main(putanja):
    
    vec_sabrani = []
    vec_oduzeti = []

    vec_uzete_tackeSABIRAJ = []
    vec_uzete_tackeODUZMI = []

    video = cv2.VideoCapture(putanja)
    detectedLines = {}
    
    suma = 0
    sumaSabranih = 0
    sumaOduzetih = 0

    while(video.isOpened): # while 1
        ret, frame = video.read()

        if not ret:
            break

        detectedLines['plavaLinija']  = dh.findBlueLine2(frame)
        addLine = detectedLines['plavaLinija']

        detectedLines['zelenaLinija'] = dh.findGreenLine2(frame)
        subLine = detectedLines['zelenaLinija']

        #cv2.line(frame, (detectedLines['plavaLinija'][0][0], detectedLines['plavaLinija'][0][1]), (detectedLines['plavaLinija'][1][0], detectedLines['plavaLinija'][1][1]), (255, 255, 235), 2)
        #cv2.line(frame, (detectedLines['zelenaLinija'][0][0], detectedLines['zelenaLinija'][0][1]), (detectedLines['zelenaLinija'][1][0], detectedLines['zelenaLinija'][1][1]), (255, 255, 255), 2)
        
        new_frameGRAY = image_gray(frame)
        new_frameBIN = image_bin(new_frameGRAY)
        deskew(new_frameBIN)
        new_frameBIN = sl.erode(sl.dilate(new_frameBIN))
        
        image_orig, sorted_regionsPlava, sorted_regionsZelena, selectRoiX, selectRoiY, tacke_arrayPlava, tacke_arrayZelena = select_roi(frame, new_frameBIN, addLine, subLine)

        if (len(sorted_regionsPlava) > 0):

            result = ann2.predict(np.array(prepare_for_ann(sorted_regionsPlava), np.float32))
            rezultat=[]
            rezultat = display_result(result)

            for r in rezultat:

                for tackaPlava in tacke_arrayPlava:

                    vec_uzete_tackeSABIRAJ.append(tackaPlava)
                    vec_sabrani.append(r)

                    if (len(vec_sabrani) >= 2 and len(vec_uzete_tackeSABIRAJ) >= 2):
                        if(vec_sabrani[len(vec_sabrani)-2] == vec_sabrani[len(vec_sabrani)-1]):
                            if (distance2D(vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ)-2], vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ)-1]) <= 3.5):
                                print('+')
                        else:
                            sumaSabranih += r
                            print('SABIRAM BROJ: ', r)
                    else:
                        print('SABIRAM BROJ: ', r)
                        sumaSabranih += r
                

        if (len(sorted_regionsZelena) > 0):

            result = ann2.predict(np.array(prepare_for_ann(sorted_regionsZelena), np.float32))
            rezultat=[]
            rezultat = display_result(result)

            for r in rezultat:
                for tackaZelena in tacke_arrayZelena:

                    vec_uzete_tackeODUZMI.append(tackaZelena)
                    vec_oduzeti.append(r)
                    
                    if (len(vec_oduzeti) >= 2 and len(vec_uzete_tackeODUZMI) >= 2):
                        if (vec_oduzeti[len(vec_oduzeti)-2] == vec_oduzeti[len(vec_oduzeti)-1]):
                            if (distance2D(vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI)-2], vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI)-1]) <= 3.5):

                                print('-')
                        else:
                            sumaOduzetih += r
                            print('ODUZIMAM BROJ: ', r)
                    else:
                        print('ODUZIMAM BROJ: ', r)
                        sumaOduzetih += r

        suma = sumaSabranih - sumaOduzetih

        #cv2.imshow('Frame program', image_orig)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    return suma
       


