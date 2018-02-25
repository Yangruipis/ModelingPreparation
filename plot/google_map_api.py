# -*- coding:utf-8 -*-

import Image
import urllib
import numpy as np
from cStringIO import StringIO
import matplotlib.pyplot as plt
from PIL import Image

def Gmap(centerLat,centerLon,zoomS,pixelS,size,dark,saveAddress):
    # get the map .png of your interesting area
    url = 'http://maps.googleapis.com/maps/api/staticmap?sensor=false'\
    +'&size='+str(size)+'x'+str(size)+'&center='+str(centerLat)+','\
    +str(centerLon)+'&zoom='+str(zoomS)+'&scale='+str(pixelS)\
    +'&maptype=terrain' # satellite 卫星图
    if dark==True:
        url = url+'&style=feature:all|element:all|saturation:-10|lightness:20'
        print url
        # 由于缺少api key，直接手动保存到本地即可

        # buffer = StringIO(urllib.urlopen(url).read())
        # image = Image.open(buffer)
        # if saveAddress:
        #    image.save(saveAddress)
        # else:
        #    image.show()

def latLonToPixelXY(lat,lon,zoomS):
    mapW = 256*2**zoomS+0.0
    mapH = 256*2**zoomS+0.0
    x = (lon+180)*(mapW/360)# get x value
    latRad = lat*np.pi/180# convert from degrees to radians
    mercN = np.log(np.tan((np.pi/4)+(latRad/2)))# get y value
    y = (mapH/2)-(mapW*mercN/(2*np.pi))
    return x,y

def sample(lis,amount):
    # 作图样本太多时用于抽样
    import random
    num_set = set()
    while(len(num_set)<amount):
        rnd = random.randint(0,len(lis)-1)
        num_set.add(rnd)
    return [lis[i] for i in num_set]

if __name__ == '__main__':
    # 输入区域坐标。以及大小
    filename = "./datafile/beijing.png"
    centerLat,centerLon = (23.157105,113.256031); scale = 12; pixelS = 2; size = 640
    #Gmap(centerLat,centerLon,scale,pixelS,size,True,'my_map.png')

    centX, centY = latLonToPixelXY(centerLat, centerLon, scale)
    data = {'23.157105_113.256031': 5,
            '23.190583_113.256031': 1,
            '23.110411_113.256031': 16}

    M = {}
    for x, y in data.items():
        lat, lon = map(float, x.split('_'))
        i, j = latLonToPixelXY(float(lat), float(lon), scale)
        i, j = size * pixelS / 2 + i - centX, size * pixelS / 2 - (j - centY)
        M[(i, j)] = y
    from PIL import Image
    im = Image.open(filename)#np.flipud(plt.imread(filename))
    ax = plt.subplot(111)
    ax.imshow(im)
    for x, y in M:
        ax.scatter(x, y, s=100 * M[(x, y)], facecolor='RoyalBlue', lw=1, alpha=0.7)
    ax.set_xlim(0, size * pixelS)
    ax.set_ylim(0, size * pixelS)
    plt.axis('off')
    # plt.show()
    plt.savefig('./datafile/beijingDots.png')



