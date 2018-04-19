import numpy as np
from numpy.random import randint, uniform, shuffle
from numpy import concatenate, array, mean, append, empty, linspace, tile, stack, copy
import random
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import random

from opti_alg_Mine_2 import*



#---------------------------------------------------set your parameters here---------------------------------------------------------#

BestFoMSoFar = 0.0000000000001                  #### Do not edit this####
target_wavelength=520                           #design wavelength 
n_glass=1.459                                   #refractive index of substrate material 
n_tio2=4.69                                     #refractive index of nanoantennae material
NumberOfFoodSources = 150                       #larger this number, better the quality of initial FoodSource
cyl_height = 500                                #height of nanoantennae
lateral_period=400                              #period of cell in lateral direction
grating_period=678.81                           #grating period of cell
min_diameter = 50.0                             #minimum diameter of nanoantennae
min_distance = 50.0                             #minimum distance between two nanoantennae
TrialsBeforeBeeIsTired = 400                    #larger the number more time Employed and onlooker bees will spend in finding a better food source
num_EmployedBees = 150                          #number of employed bees(note: keep Employed and Onlooker bees same number).
num_OnLookerBees =150                           #number of onlooker bees
num_ScoutBees = 0                               #number of scout bees (note: start with a value of zero)
a = -0.1250                                     #a and b dictate how wide the memetic search should be. "a" must be negative and opposite of "b"
b = 0.1250                                      #a and b dictate how wide the memetic search should be, "b" must be positive and opposite of "a"
Epsilon = 0.001                                 #stopping criteria of Memetic Phase, Lower the number, longer the search, more chances of finding a better value
Psi = 0.681                                     #golden ratio
Pr = 0.8                                        #probability to update the position
C = 1.5                                         #a non negative constant for position update
FamilySize = 2                                  #number of members in the Family
NumberOfGenerations = 100                       #number of generations(cycles) of Food Search of Emp Bee, Onllok Bee, Scout Bee and Memetic phase

##---------------------------------------------------------------------------------------------------------------------------------##


def sq_distance_mod(x0,y0,x1,y1,x_period,y_period):
    """squared distance between two points in a 2d periodic structure"""
    dx = min((x0 - x1) % x_period, (x1 - x0) % x_period)
    dy = min((y0 - y1) % y_period, (y1 - y0) % y_period)
    return dx*dx + dy*dy

def distance_mod(x0,x1,period):
    """1d version - distance between two points in a periodic structure"""
    return min((x0 - x1) % period, (x1 - x0) % period)

def validate(xyrra_list, grating_period, lateral_period, print_details=False, similar_to=None, how_similar=None):
    """ make sure the structure can be fabricated, doesn't self-intersect, etc.
        If similar_to is provided, it's an xyrra_list describing a configuration
    that we need to resemble, and how_similar should be a factor like 0.02
    meaning the radii etc. should change by less than 2%."""
    if np.array(xyrra_list[:,[2,3]]).min() < min_diameter/2:
        if print_details:
            print('a diameter is too small')
        return False

    # Check that no two shapes are excessively close
    # points_list[i][j,k] is the x (if k=0) or y (if k=1) coordinate
    # of the j'th point on the border of the i'th shape
    points_per_ellipse = 100
    num_ellipses = xyrra_list.shape[0]
    points_arrays = []
    for i in range(num_ellipses):
        points_arrays.append(ellipse_pts(*xyrra_list[i,:], num_points=points_per_ellipse))
    
    # first, check each shape against its own periodic replicas, in the
    # smaller (y) direction. I assume that shapes will not be big enough to 
    # approach their periodic replicas
    for i in range(num_ellipses):
        i_pt_list = points_arrays[i]
        j_pt_list = i_pt_list.copy()
        j_pt_list[:,1] += lateral_period
        for i_pt in i_pt_list:
            for j_pt in j_pt_list:
                if (i_pt[0] - j_pt[0])**2 + (i_pt[1] - j_pt[1])**2 < min_distance**2:
                    if print_details:
                        print('too close, between ellipse', i, 'and its periodic replica')
                        print(i_pt)
                        print(j_pt)
                    return False
    
    for i in range(1,num_ellipses):
        i_pt_list = points_arrays[i]
        for j in range(i):
            j_pt_list = points_arrays[j]
            for i_pt in i_pt_list:
                for j_pt in j_pt_list:
                    if sq_distance_mod(i_pt[0], i_pt[1], j_pt[0], j_pt[1],
                                       grating_period, lateral_period) < min_distance**2:
                        if print_details:
                            print('too close, between ellipse', j, 'and', i)
                            print(i_pt)
                            print(j_pt)
                        return False
    if similar_to is not None:
        for i in range(num_ellipses):
            if max(abs(xyrra_list[i, 2:4] - similar_to[i, 2:4]) / similar_to[i, 2:4]) > how_similar:
                if print_details:
                    print('A radius of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,0], similar_to[i,0], grating_period) > how_similar * grating_period:
                if print_details:
                    print('x-coordinate of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,1], similar_to[i,1], lateral_period) > how_similar * lateral_period:
                if print_details:
                    print('y-coordinate of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,4], similar_to[i,4], 2*pi) > how_similar * (2*pi):
                if print_details:
                    print('rotation of ellipse', i, 'changed too much')
                return False
    return True

def ellipse_pts(x_center, y_center, r_x, r_y, angle, num_points=80):
    """return a list of (x,y) coordinates of points on an ellipse, in CCW order"""
    xy_list = empty(shape=(num_points,2))
    theta_list = linspace(0,2*pi,num=num_points, endpoint=False)
    for i,theta in enumerate(theta_list):
        dx0 = r_x * math.cos(theta)
        dy0 = r_y * math.sin(theta)
        xy_list[i,0] = x_center + dx0 * math.cos(angle) - dy0 * math.sin(angle)
        xy_list[i,1] = y_center + dx0 * math.sin(angle) + dy0 * math.cos(angle)
    return xy_list



def FoM(FamilyOfGrating, lateral_period, grating_period, cyl_height):
    my_grating =Grating(lateral_period=lateral_period*nm, grating_period= grating_period*nm, cyl_height=cyl_height*nm, n_glass=n_glass, n_tio2=n_tio2, xyrra_list_in_nm_deg=np.array(FamilyOfGrating), data=None)    
    my_grating.get_xyrra_list(units='nm,deg', replicas=10)
    my_grating.xyrra_list_in_nm_deg
    my_grating.copy
    my_grating.run_lua(target_wavelength=target_wavelength*nm,subfolder='temp')
    my_grating.run_lua_initiate(target_wavelength=target_wavelength*nm,subfolder='temp')
    global BestFoMSoFar 
    FigureOfMerit = round(my_grating.run_lua(subfolder='temp',target_wavelength=target_wavelength*nm,numG=100), 8)
    if BestFoMSoFar == 0.000000000001 and FigureOfMerit < BestFoMSoFar:
        print(FigureOfMerit," .... and Best FoM so far ",BestFoMSoFar)
    elif FigureOfMerit > BestFoMSoFar:
        BestFoMSoFar = FigureOfMerit
        print(FigureOfMerit," .... and Best FoM so far ",BestFoMSoFar)
        np.save("BestFamily.txt",FamilyOfGrating)
    else:
        print(FigureOfMerit," .... and Best FoM so far ",BestFoMSoFar)
    return [FamilyOfGrating, FigureOfMerit]

    
class MeABC:

    def __init__(self, StartingGuess=None, xmin=-0.000001, ymin=-0.000001,
                 xmax=grating_period, ymax=lateral_period ,
                 majorAxisMin=min_diameter, MajorAxisMax=150.000000001,
                 minorAxisMin=min_diameter, MinorAxisMax=150.000000001,angleMin=0.0001,
                 angleMax=359.0001, grating_period = grating_period, lateral_period = lateral_period, cyl_height=cyl_height,
                 FamilySize=FamilySize, IndividualsInSingleGen = NumberOfFoodSources, NumberOfGenerations=NumberOfGenerations,
                 num_EmployedBees=num_EmployedBees, num_OnLookerBees= num_OnLookerBees, num_ScoutBees = num_ScoutBees, TrialsBeforeBeeIsTired=TrialsBeforeBeeIsTired,
                 a=a,b=b,Psi=Psi,
                 Pr=Pr, C = C):
        self.StartingGuess = StartingGuess ## Not implemented
        self.IndividualsInSingleGen = IndividualsInSingleGen
        self.NumberOfGenerations = NumberOfGenerations
        self.FamilySize = FamilySize
        self.grating_period = grating_period
        self.lateral_period = lateral_period
        self.cyl_height = cyl_height
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.majorAxisMin = majorAxisMin
        self.MajorAxisMax = MajorAxisMax
        self.minorAxisMin = minorAxisMin
        self.MinorAxisMax = MinorAxisMax
        self.angleMin = angleMin
        self.angleMax = angleMax
        self.num_EmployedBees = num_EmployedBees
        self.num_OnLookerBees = num_OnLookerBees
        self.num_ScoutBees = num_ScoutBees
        self.TrialsBeforeBeeIsTired = TrialsBeforeBeeIsTired
        self.a = a
        self.b = b
        self.Psi = Psi
        self.Pr = Pr
        self.C = C

    def GetNumEmpBees(self):
        return self.num_EmployedBees

    def GetNumOnlookerBees(self):
        return self.num_OnLookerBees

    def GetNumScoutBees(self):
        return self.num_ScoutBees

    def GetNumGenerations(self):
        return self.NumberOfGenerations

    def IntializeEmployedBeePopulation(self, num_EmployedBees):
        return np.zeros(shape=(num_EmployedBees, self.FamilySize, 6, 1))

    def InitializeOnlookerBeePopulation(self, num_OnLookerBees):
        return np.zeros(shape=(num_OnLookerBees, self.FamilySize, 6, 1))

    def InitializeScoutBeePopulation(self, num_ScoutBees):
        return np.zeros(shape=(num_ScoutBees, self.FamilySize, 6, 1))
        
    def GetTrialsBeforeBeeIsTired(self):
        return self.TrialsBeforeBeeIsTired
        

    def MakeInitialFoodSources(self, FoodSource = None):
        if FoodSource == None:
            FoodSource = np.zeros(shape=(NumberOfFoodSources, self.FamilySize, 6, 1))
            FamilyOfGrating = np.zeros(shape=(self.FamilySize,5,1))
            
            for i in range(NumberOfFoodSources):
                grating_period = self.grating_period
                lateral_period = self.lateral_period
                cyl_height = self.cyl_height
                validated = False
                while(not validated):
                    for a in range(self.FamilySize):
                        X = uniform(low=self.xmin,high=grating_period,size=1)
                        Y = uniform(low=self.ymin,high=lateral_period,size=1)
                        two_axis= [uniform(low=self.majorAxisMin,high=self.MajorAxisMax,size=1), uniform(low=self.minorAxisMin,high=self.MinorAxisMax,size=1)]
                        r_major= max(two_axis)
                        r_minor= min(two_axis)
                        angle = uniform(low=self.angleMin,high=self.angleMax,size=1)
                        FamilyOfGrating[a]=[X,Y,r_major,r_minor,angle]
                        FoodSource[i][a]=np.array([X,Y,r_major,r_minor,angle, BestFoMSoFar]).reshape(6,1)
                    validated = validate(FamilyOfGrating, grating_period, lateral_period, print_details=False)
                    if validated:
                        print("Successfully validated the Family")
                newFoM = FoM(np.array(FamilyOfGrating), lateral_period,grating_period,cyl_height)[1]
                for z in range(self.FamilySize):
                    FoodSource[i][z][5]=newFoM
                if i != 0:
                    print("Found %d Families out of %d families.\n"%(i+1,NumberOfFoodSources))
                else:
                    print("Found 1st Family out of %d families.\n"%NumberOfFoodSources)
            return FoodSource
        else:
            return FoodSource
        
    def FindFoMForAFoodPatch(self, FoodPatch):
        StartingFamily = np.array(FoodPatch[:,0:5])
        lateral_period = self.lateral_period
        grating_period = self.grating_period
        cyl_height = self.cyl_height
        return FoM(np.array(StartingFamily),lateral_period,grating_period,cyl_height)
        

    def FindBestFoodSource(self, FoodSource):
        BestFoM = max(FoodSource[:,0,5,0])
        print("Best FoM for the food source is ",BestFoM)
        for each in FoodSource:
            if each[0][5][0]==BestFoM:
                print("Found the Best Food Patch")
                return each
        
    def MemeticSearchPhase(self, a, b, Epsilon, Psi, BestFoodPatch):
        assert a<b,"a must be less than b"
        BestFoM = BestFoodPatch[0][5][0]
        StartingFamily=BestFoodPatch[:,(0,5),0]
        Validated = False
        counter = 1
        lateral_period = self.lateral_period
        grating_period = self.grating_period
        cyl_height = self.cyl_height

        while(abs(a-b)>Epsilon):
            print(a,b)
            print("Starting pass number %d of Memetic Phase"%counter)
            F1 = b-(b-a)*Psi
            F2 = a+(b-a)*Psi
            Bounded = False
            InnerCounter = 1
            while(not Bounded and InnerCounter <= 2):
                xnew1 = BestFoodPatch+BestFoodPatch*F1
                xnew2 = BestFoodPatch+BestFoodPatch*F2
                FoM_xnew1 = FoM(np.array(xnew1[:,0:5]),lateral_period,grating_period,cyl_height)[1]
                for z in range(self.FamilySize):
                    xnew1[z][5]=FoM_xnew1                
                FoM_xnew2 = FoM(np.array(xnew2[:,0:5]),lateral_period,grating_period,cyl_height)[1]
                for z in range(self.FamilySize):
                    xnew2[z][5]=FoM_xnew2
                if FoM_xnew1 > FoM_xnew2:
                    b = F2
                    if FoM_xnew1 > BestFoM:
                        BestFoodPatch = xnew1
                        BestFoM = FoM_xnew1

                else:
                    a = F1
                    if FoM_xnew2 > BestFoM:
                        BestFoodPatch = xnew2
                        BestFoM = FoM_xnew2

                Bounded = self.BoundsCheck(BestFoodPatch)
                if Bounded:
                    print("Bounds Check succeeded")
                else:
                    print("Bounds check failed")
                InnerCounter += 1
            print("Ended pass number %d of Memetic Phase"%counter)
            counter += 1
            StartingFamily=BestFoodPatch[:,0:5]
            Validated = validate(np.array(StartingFamily),grating_period,lateral_period)
        print("Best Food Patch is ",BestFoodPatch)
        return BestFoodPatch

    def BoundsCheck(self, FoodPatch): ##This part/function is not a smart programming. Can definitely be made better!!
        TestFlag = False
        for i in range(self.FamilySize):
            if (FoodPatch[i][2][0]>=self.majorAxisMin and FoodPatch[i][2][0]<=self.MajorAxisMax):
                if (FoodPatch[i][3][0]>=self.minorAxisMin and FoodPatch[i][3][0]<=self.MinorAxisMax):
                    if (FoodPatch[i][4][0]>=self.angleMin and FoodPatch[i][3][0]<=self.angleMax):
                        TestFlag = True
                    else:
                        TestFlag = False
                        break
                else:
                    TestFlag = False
                    break
            else:
                TestFlag = False
                break
        return TestFlag

    def Find_N_BestFoodPatches(self, FoodSource, NumberOfFoodPatches = num_EmployedBees):
        SortedFoMs=np.array(sorted((np.array(FoodSource[:,0,5,0])).flatten(),reverse=True)).reshape(len(FoodSource),1)[0:NumberOfFoodPatches,]
        ReturnedFoodSource = np.zeros(shape=(NumberOfFoodPatches,self.FamilySize,6,1))
        Index = 0
        while(Index<NumberOfFoodPatches):
            for each in FoodSource:
                if each[0][5][0] in SortedFoMs:
                    ReturnedFoodSource[Index]=each
                    Index += 1
        return ReturnedFoodSource 

                                
    def EmployedBee(self, FoodPatch, BestFoodPatch, FoodSource, Index):

        Validated = False
        while(not Validated):
            Bounds = False
            while(not Bounds):
                for a in range(self.FamilySize):
                    Phi = uniform(low= -1.000000, high = 1.000000, size=1)
                    Psi = uniform(low= 0.0000000, high = self.C, size =1)
                    randomMemberIndex = int(uniform(low=0,high=len(FoodSource),size=1))
                    while(randomMemberIndex == Index):
                        randomMemberIndex = int(uniform(low=0,high=len(FoodSource),size=1))
                    Update_Chance = uniform(low=0,high=1,size=1)
                    for i in range(5):
                        if Update_Chance < self.Pr:
                            FoodPatch[a][i]=FoodPatch[a][i]+(Phi*(FoodPatch[a][i]-FoodSource[randomMemberIndex][a][i]))+(Psi*(BestFoodPatch[a][i]-FoodPatch[a][i]))
                            
                Bounds = self.BoundsCheck(FoodPatch)
            StartingFamily = FoodPatch[:,0:5]
            grating_period = self.grating_period
            lateral_period = self.lateral_period
            cyl_height = self.cyl_height
            Validated = validate(np.array(StartingFamily),grating_period, lateral_period)
        NewFoM=FoM(np.array(StartingFamily),lateral_period,grating_period,cyl_height)[1]
        for z in range(self.FamilySize):
            FoodPatch[z][5][0]=NewFoM
        return FoodPatch
        
    def BestFoodForOnlookerBees(self, FoodSource):
        BestFoodPatch = np.zeros(shape=(self.FamilySize,6,1))
        BestProbability = 0.00001
        SumOfFitness = sum(FoodSource[:,0,5,0])
        MinProb = min(FoodSource[:,0,5,0])/SumOfFitness
        MaxProb = max(FoodSource[:,0,5,0])/SumOfFitness 
        selected = False
        while (not selected):
            for FoodPatch in FoodSource:
                Probability = FoodPatch[0][5][0] / SumOfFitness       
                ChanceToSelect = uniform(low=MinProb, high=MaxProb, size=1)
                if Probability < ChanceToSelect:
                    if Probability > BestProbability:
                        selected = True
                        BestFoodPatch = FoodPatch 
                        BestProbability = Probability
                        break
                else:
                    continue
        if selected:
            return BestFoodPatch
                

    def OnLookerBee(self, BestFoodPatch, FoodSource, FoodPatch, Index):

        Validated = False
        while(not Validated):

            Bounds = False
            while(not Bounds):
                for a in range(self.FamilySize):
                    for i in range(5):
                        Phi = uniform(low= -1.000000, high = 1.000000, size=1)
                        Psi = uniform(low= 0.0000000, high = self.C, size =1)
                        randomMemberIndex = int(uniform(low=0,high=len(FoodSource),size=1))
                        while(randomMemberIndex == Index):
                            randomMemberIndex = int(uniform(low=0,high=len(FoodSource),size=1))
                        Update_Chance = uniform(low=0,high=1,size=1)
                        if Update_Chance < self.Pr:
                            FoodPatch[a][i]=FoodPatch[a][i]+(Phi*(FoodPatch[a][i]-FoodSource[randomMemberIndex][a][i]))+(Psi*(BestFoodPatch[a][i]-FoodPatch[a][i]))
                            
                Bounds = self.BoundsCheck(FoodPatch)
            StartingFamily = FoodPatch[:,0:5]
            grating_period = self.grating_period
            lateral_period = self.lateral_period
            cyl_height = self.cyl_height
            Validated = validate(np.array(StartingFamily), grating_period, lateral_period)
        NewFoM=FoM(np.array(StartingFamily),lateral_period,grating_period,cyl_height)[1]
        for z in range(self.FamilySize):
            FoodPatch[z][5][0]=NewFoM

        return [FoodPatch, FoodSource]
        

    def ScoutBees(self, FoodPatch):
        SelectedFood = np.zeros(shape=(self.FamilySize,6,1))
        Family=np.zeros(shape=(self.FamilySize,5,1))
        Validated = False
        grating_period = self.grating_period
        lateral_period = self.lateral_period
        cyl_height = self.cyl_height
        while(not Validated):
            for i in range(self.FamilySize):
                X = uniform(low=self.xmin,high=self.xmax,size=1)
                Y = uniform(low=self.ymin,high=self.ymax,size=1)
                two_axis= [uniform(low=self.majorAxisMin,high=self.MajorAxisMax,size=1), uniform(low=self.minorAxisMin,high=self.MinorAxisMax,size=1)]
                r_major= max(two_axis)
                r_minor= min(two_axis)
                angle = uniform(low=self.angleMin,high=self.angleMax,size=1)
                Family[i]=np.array([X,Y,r_major,r_minor,angle]).reshape(5,1)

                SelectedFood[i] =np.array([X,Y,r_major,r_minor,angle, 0.000001]).reshape(6,1)

            Validated = validate(np.array(Family),grating_period, lateral_period)
        newFoM = FoM(np.array(Family), lateral_period, grating_period, cyl_height)[1]
        for z in range(self.FamilySize):
            SelectedFood[z][5][0] = newFoM
        return SelectedFood


##******************* Main Program *******************

A = MeABC()
NumEmpBees = A.GetNumEmpBees()
EmployedBees = A.IntializeEmployedBeePopulation(NumEmpBees)
NumOnlookerBees = A.GetNumOnlookerBees()
OnlookerBees = A.InitializeOnlookerBeePopulation(NumOnlookerBees)
NumScoutBees = A.GetNumScoutBees()
ScoutBees = A.InitializeScoutBeePopulation(NumScoutBees)
FoodSource = A.MakeInitialFoodSources()
np.save("FoodSources.txt",FoodSource)
FoodSource=np.load("FoodSources.txt.npy")
NumberOfGenerations = A.GetNumGenerations()
BestFoodPatch = A.FindBestFoodSource(FoodSource)
ScoutBeeID = NumScoutBees
ScoutBeeIndex = []
OriginalFoodSource = FoodSource
for i in range(NumberOfGenerations):
    print("This is Generation number %d"%(i+1))

##### Employer Bee Phase #####

    FoodSource = A.Find_N_BestFoodPatches(FoodSource)
    BestFoodPatch = BestFoodPatch
    EmpBeeID = 1
    StartingFoodID = 0
    for k in range(len(FoodSource)):
        print("Deploying Employe Bee number %d"%EmpBeeID)
        StartingFoM = FoodSource[k][0][5][0]
        TrialCounter = 1
        TrialsBeforeBeeIsTired = A.GetTrialsBeforeBeeIsTired()
        while(TrialsBeforeBeeIsTired):
            print("This is Trial number %d for Employe Bee %d"%(TrialCounter,EmpBeeID))
            FoodSource[k]=A.EmployedBee(FoodSource[k],BestFoodPatch,FoodSource, k)
            newFoM = FoodSource[k][0][5][0]
            if newFoM > StartingFoM:
                EmployedBees[EmpBeeID-1]=FoodSource[k]
                print("The empoyed Bee ID %d found a better food source and the FoM increased from %.7f to %.7f"%(EmpBeeID,StartingFoM,newFoM))
                break
            TrialsBeforeBeeIsTired -=1
            if TrialsBeforeBeeIsTired == 0:
                ScoutBeeID += 1
                ScoutBeeIndex.append(k)
            TrialCounter += 1
        EmpBeeID += 1
    FileHandle="EmployeeBeePhase-"+str(i)+"-Solution.txt"
    np.save(FileHandle, FoodSource)

###### OnlookerBeePhase #######
    FoodSource = EmployedBees
    BestFoodPatch = A.FindBestFoodSource(FoodSource)
    StartingFoM = BestFoodPatch[0][5][0]
    OnlookBeeID = 1
    BestFood = A.BestFoodForOnlookerBees(FoodSource)

    for each in range(len(FoodSource)):
        eachFood = FoodSource[each]
        print("Deploying Onlooker Bee number %d" % OnlookBeeID)
        TrialCounter = 1
        TrialsBeforeBeeIsTired = A.GetTrialsBeforeBeeIsTired()
        while(TrialsBeforeBeeIsTired):
            print("This is Trial number %d for Onlooker Bee %d"%(TrialCounter,OnlookBeeID))
            FoodInfo = A.OnLookerBee(BestFood,FoodSource, eachFood, each)
            FoodPatch = FoodInfo[0]
            newFoM = FoodPatch[0][5][0]
            if newFoM > StartingFoM:
                FoodSource[each] = FoodPatch
                OnlookerBees[OnlookBeeID-1]=FoodPatch
                FoodSource = FoodInfo[1]
                print("The Onlooker Bee ID %d found a better food source and the FoM increased from %.7f to %.7f"%(OnlookBeeID,StartingFoM,newFoM))
                break
            TrialsBeforeBeeIsTired -=1
            if TrialsBeforeBeeIsTired == 0:
                ScoutBeeID += 1
                ScoutBeeIndex.append(each)
            TrialCounter += 1
        OnlookBeeID += 1
    FileHandle="OnlookerBeePhase-"+str(i)+"-Solution.txt"
    np.save(FileHandle, FoodSource)
#### ScoutBeePhase 
    
    while(ScoutBeeID):
        print("Deploying a scout Bee")
        for l in ScoutBeeIndex:
            FoodSource[l]=A.ScoutBees(FoodSource[l])
        print("Scout Bee found a food source")
        ScoutBeeID -= 1
    FileHandle="ScoutBeePhase-"+str(i)+"-Solution.txt"
    np.save(FileHandle, FoodSource)
                
##### Memetic Search Phase
    BestFoodPatch = A.FindBestFoodSource(FoodSource)
    StartingFoM=BestFoodPatch[0][5][0]
    Index = 0
    for j in range(len(FoodSource)):
        if (np.array(FoodSource[j])==np.array(BestFoodPatch)).all():
            Index = j
            break
    BestFoodPatch = A.MemeticSearchPhase(a, b, Epsilon, Psi, BestFoodPatch)    
    FoodSource[Index]=BestFoodPatch
    EndingFoM=BestFoodPatch[0][5][0]
    FileHandle="BestFoodPatch-"+str(i)+"-Solution.txt"
    np.save(FileHandle, BestFoodPatch)
    if EndingFoM >StartingFoM:
        print("Memetic Search Phase was successfull in finding a better food source")
    else:
        print("Memetic Search phase failed")
    BestFoMSoFar = BestFoodPatch[0][5][0]
    print("Best FoM so far after %d generation is %.7f"%((i+1),BestFoMSoFar))
    
np.save("FinalFoodSource.txt",FoodSource)    



    
            
            
        
        

    
                
            
        
                
                
                
                
        




