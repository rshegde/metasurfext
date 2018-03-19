from numpy.random import randint, uniform, shuffle
from numpy import concatenate, array, mean, append, empty, linspace, tile, stack, copy
import random

from grating import*

FoM_Dict_ByGeneration={}
FoundFamilies = 0
BestFoMSoFar = 0.000000000001
BestFamily = []

#-------------------------------------------Input--------------------------------------------------
target_wavelength=580*nm
min_diameter = 100
min_distance = 100
lateral_period=400.00
grating_period=902.31

FamilySize=3
NumberOfGenerations=50
IndividualsInSingleGen=2
mutationRate=0.07
CrossingOverRate=0.95
#--------------------------------------------------------------------------------------------------

def sq_distance_mod(x0,y0,x1,y1,x_period,y_period):
    """squared distance between two points in a 2d periodic structure"""
    dx = min((x0 - x1) % x_period, (x1 - x0) % x_period)
    dy = min((y0 - y1) % y_period, (y1 - y0) % y_period)
    return dx*dx + dy*dy

def distance_mod(x0,x1,period):
    """1d version - distance between two points in a periodic structure"""
    return min((x0 - x1) % period, (x1 - x0) % period)

def validate(xyrra_list, print_details=False, similar_to=None, how_similar=None):
    """ make sure the structure can be fabricated, doesn't self-intersect, etc.
        If similar_to is provided, it's an xyrra_list describing a configuration
    that we need to resemble, and how_similar should be a factor like 0.02
    meaning the radii etc. should change by less than 2%."""
    
    if xyrra_list[:,[2,3]].min() < min_diameter/2:
        if print_details:
            print('a diameter is too small')
        return False
#    if xyrra_list[:,3].max() > max_y_diameter/2:
#        if print_details:
#            print('a y-diameter is too big')
#        return False
    
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

#### ============= End of Helper functions ============================#####

def FoM(FamilyOfGrating):
    global BestFoMSoFar
    my_grating =Grating(lateral_period=400*nm, grating_period=grating_period*nm, cyl_height=550.0*nm, n_glass=0, n_tio2=0, xyrra_list_in_nm_deg=np.array(FamilyOfGrating), data=None)    
    my_grating.get_xyrra_list(units='nm,deg', replicas=1)
    my_grating.xyrra_list_in_nm_deg
    my_grating.copy
    my_grating.run_lua(target_wavelength=580*nm,subfolder='temp')
    my_grating.run_lua_initiate(target_wavelength=580*nm,subfolder='temp')
    FigureOfMerit = my_grating.run_lua(subfolder='temp',target_wavelength=580*nm,numG=50)
    if FigureOfMerit > BestFoMSoFar:
        BestFoMSoFar=FigureOfMerit
        BestFamily = FamilyOfGrating
        np.save("BestFamily.txt", BestFamily)
    print(FigureOfMerit," .... and Best FoM from previous generation was ",BestFoMSoFar)
    return [FamilyOfGrating, FigureOfMerit]

class GA:

    def __init__(self, StartingGuess=None, xmin=(-249.9999999), ymin=(-249.99999999),
                 xmax=grating_period+249.9999999, ymax=400+249.99999999,
                 majorAxisMin=50, MajorAxisMax=150,
                 minorAxisMin=50, MinorAxisMax=150,angleMin=(0.00001),
                 angleMax=(359.99999999),
                 FamilySize=3, IndividualsInSingleGen =900, NumberOfGenerations=50, converged=False):
        self.StartingGuess = StartingGuess
        self.IndividualsInSingleGen = IndividualsInSingleGen
        self.NumberOfGenerations = NumberOfGenerations
        self.FamilySize = FamilySize
        self.converged = converged
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

    def Individual(self):

        two_axis= [uniform(low=self.majorAxisMin,high=self.MajorAxisMax,size=1), uniform(low=self.minorAxisMin,high=self.MinorAxisMax,size=1)]
        r_major= max(two_axis)
        r_minor= min(two_axis)
        Indiv = array([uniform(low=self.xmin,high=self.xmax,size=1),uniform(low=self.ymin,high=self.ymax,size=1),r_major, r_minor, uniform(low=self.angleMin, high=self.angleMax,size=1)])
        return Indiv

            
        


    def Family(self, Families=None):
        if Families==None:
            global FoundFamilies
            Individual_1=self.Individual()
            Individual_2=self.Individual()
            Individual_3=self.Individual()
         #   Individual_4=self.Individual()
         #   Individual_5=self.Individual()
            
            StartingFamily = array([Individual_1,Individual_2,Individual_3])

            while (not validate(StartingFamily, print_details=True)):
                Individual_1=self.Individual()
                Individual_2=self.Individual()
                Individual_3=self.Individual()
              #  Individual_4=self.Individual()
             #   Individual_5=self.Individual()
            
                StartingFamily = array([Individual_1,Individual_2,Individual_3])

            FoundFamilies += 1
            print("So far we have found %d families \n"%FoundFamilies)

        else:
            print("\n")
            FoundFamilies += 1
            print("Using the Starting Family given and we have found %d families\n"%FoundFamilies)
            StartingFamily = self.StartingGuess

        return array(StartingFamily)
            
    
        
    def MakeThePopulationOfFamilies(self, StartingFamily=None):
        EntirePopulation = []
        if StartingFamily == None:            
            for i in range(0,self.IndividualsInSingleGen,self.FamilySize):
                EntirePopulation.append(self.Family())
        else:
            for i in range(0,int((self.IndividualsInSingleGen//self.FamilySize)*0.15)):
                EntirePopulation.append(self.Family(StartingFamily))
            for i in range(0,int((self.IndividualsInSingleGen//self.FamilySize)*0.85)):
                EntirePopulation.append(self.Family())
            
        return EntirePopulation
        


    def MutateIndividualsInFamily(self, Family, mutationRate = 0.07, maxTrials=200):
        print("Inside Mutate function")
        count = 0
        while((not validate(Family, print_details=True) and count<=maxTrials//20) or count == 0 ):
            print("This is %d Outer loop of mutation"%count)
            for eachMember in Family:
                CopyEachMember = copy(eachMember)
                ScoreToMutate = random.random()
                if ScoreToMutate < mutationRate:
                        Done = False
                        counter = 1
                        while(not Done and counter<=maxTrials//50):
                            print("This is %d trial to Mutate"%counter)
                            counter += 1
                            for i in range(0,5):
                                Percentage_Change= random.uniform(0.00001,0.12000)
                                PlusOrMinus = random.choice('+-')
                                if PlusOrMinus == '+':
                                    eachMember[i]=CopyEachMember[i]
                                    eachMember[i] = eachMember[i]+eachMember[i]*Percentage_Change
                                else:
                                    eachMember[i]=CopyEachMember[i]
                                    eachMember[i] = eachMember[i]-eachMember[i]*Percentage_Change
                            if eachMember[0]>=self.xmin and eachMember[0]<=self.xmax:
                                if eachMember[1]>=self.ymin and eachMember[1]<=self.ymax:
                                    if eachMember[2]>=self.majorAxisMin and eachMember[2]<=self.MajorAxisMax:
                                        if eachMember[3]>=self.minorAxisMin and eachMember[3]<=self.MinorAxisMax:
                                            Done = True
                            if counter ==maxTrials:
                                print("Aborting mutation!! This is a possible sign of global convergence or getting trapped in a local minima")
            count += 1
                    
        return array(Family)
                    

    def CrossingOfIndividualsOfDifferentFamilies(self, Individual_1, Individual_2, CrossingOverRate=CrossingOverRate):
        print("Starting The crossover")
        Child_1=copy(Individual_1)
        Child_2=copy(Individual_2)
        
        if random.random()<CrossingOverRate:
            Child_1[0]=Individual_2[0]
            Child_1[1]=Individual_2[1]
            Child_2[0]=Individual_1[0]
            Child_2[1]=Individual_1[1]
        if random.random()<CrossingOverRate:
            Child_1[2]=Individual_2[2]
            Child_1[3]=Individual_2[3]
            Child_2[2]=Individual_1[2]
            Child_2[3]=Individual_1[3]
        if random.random()<CrossingOverRate:
            Child_1[4]=Individual_2[4]
            Child_2[4]=Individual_1[4]

        print("Crossover Done")
            
        return [Child_1,Child_2]
                
    def Average_N_Max_FoM_Population(self, Population):
        FullFoM_List = ([FoM(Family) for Family in Population])
        AverageFoMOfPopulation = mean([each[1] for each in FullFoM_List])
        MaxFoMOfPopulation =max([each2[1] for each2 in FullFoM_List])
        MaxFoM_Family = ([each3 for each3 in FullFoM_List if each3[1]==MaxFoMOfPopulation])        
        return [AverageFoMOfPopulation,MaxFoM_Family]

    def show_config(self, xyrra_list):
        plt.figure()
        plt.xlim(-grating_period , grating_period )
        plt.ylim(-lateral_period , lateral_period )
        for x,y,rx,ry,a in xyrra_list:
            circle = matplotlib.patches.Ellipse((x , y),
                                                 2*rx, 2*ry ,
                                                 angle=a,
                                                 color='k', alpha=0.5)
            plt.gcf().gca().add_artist(circle)
        rect = matplotlib.patches.Rectangle((-grating_period/2 ,-lateral_period/2 ),
                                            grating_period , lateral_period,
                                            facecolor='none', linestyle='dashed',
                                            linewidth=2, edgecolor='red')
        plt.gcf().gca().add_artist(rect)
        plt.gcf().gca().set_aspect('equal')
            
    def GA_core(self):
        global BestFoMSoFar
        Population = self.MakeThePopulationOfFamilies(self.StartingGuess)
        np.save("StartingPopulation.txt",Population)
        Population = np.load("StartingPopulation.txt.npy")
##        AverageFoM = self.Average_N_Max_FoM_Population(Population)
##        FoM_Dict_ByGeneration[1]=AverageFoM
##        print("Average FoM of the Population after 1st Generation is:- ",AverageFoM)

        for GenNumber in range(self.NumberOfGenerations):
            New_Mutated_Population = [self.MutateIndividualsInFamily(Family) for Family in Population]
            FoM_List_Full=[]
            for eachFamily in New_Mutated_Population:
                FoM_List_Full.append(FoM(eachFamily))

            FoM_List=sorted([Figure[1] for Figure in FoM_List_Full], reverse=True)
            BestFoMSoFar=FoM_List[0]
            Top_15_Percent_FoM = FoM_List[:int(self.IndividualsInSingleGen//self.FamilySize*0.15)]
            Worst_85_Percent_FoM = FoM_List[int(self.IndividualsInSingleGen//self.FamilySize*0.15):int(self.IndividualsInSingleGen//self.FamilySize)]

            Top_15_Percent_Families = np.array([Figure[0] for Figure in FoM_List_Full if Figure[1] in Top_15_Percent_FoM])
            Worst_85_Percent_Families = np.array([Figure[0] for Figure in FoM_List_Full if Figure[1] in Worst_85_Percent_FoM])

            Random_10_Percent_Families_Index = random.sample(range(0,int(self.IndividualsInSingleGen//self.FamilySize*0.85)),int(self.IndividualsInSingleGen//self.FamilySize*0.10))
            Random_10_Percent_Families = np.array([Worst_85_Percent_Families[each] for each in Random_10_Percent_Families_Index])

            Selected_Families =[]
            for i in range(4):
                for each in Top_15_Percent_Families:
                    Selected_Families.append(each)
                for each2 in Random_10_Percent_Families:
                    Selected_Families.append(each2)
                    
            FileHandle="2_Member_Generation_number"+str(GenNumber+1)+"Top15PercentFamilies.txt"
                    
            np.save(FileHandle,np.array(Top_15_Percent_Families))
            
            Cloned_Population=np.array(Selected_Families)
            random.shuffle(Cloned_Population)

            CrossedOverFamilies =[]

            for i in range(0,len(Cloned_Population)-1,2):
                Family_1 = Cloned_Population[i]
                Family_2 = Cloned_Population[i+1]
                Family_1_copy = copy(Family_1)
                Family_2_copy = copy(Family_2)
                CrossOverCounter = 0
                while ((not validate(Family_1_copy, print_details=True)) and (not validate(Family_2_copy, print_details=True)) and CrossOverCounter < 40):
                    for j in range(self.FamilySize):
                        Family_1_copy[j],Family_2_copy[j] = self.CrossingOfIndividualsOfDifferentFamilies(Family_1[j],Family_2[j])
                        print("Validating The crossed Over Families")
                    CrossOverCounter += 1
                print("Successfully validated")        
                CrossedOverFamilies.append(Family_1_copy)
                CrossedOverFamilies.append(Family_2_copy)
            
            FileHandle="2_Member_Generation_number"+str(GenNumber+1)+"CrossedOverFamilies.txt"

            np.save(FileHandle,np.array(CrossedOverFamilies))

            Population = np.array(CrossedOverFamilies)
##            FoM_After=self.Average_N_Max_FoM_Population(Population)
##            FoM_Dict_ByGeneration[GenNumber+1]=FoM_After
##            print("Average FoM after %d generation"%int(GenNumber+1),FoM_After)
            

        return FoM_Dict_ByGeneration

#des=[[214.6283922,-23.63667232,46.64198828,40.28345675,-88.19096942],[-128.87042108,-125.9926984,103.32688395,89.08704603,-152.31985874]]
A=GA()
FoMDictionaryByGeneration=A.GA_core()
print(FoMDictionaryByGeneration)


                
                    
                    

            
                
            
            


        
        
        
    
    
