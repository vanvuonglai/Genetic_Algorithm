
import random
import pathlib
import os
from subprocess import Popen
from matplotlib import pyplot as plt
import numpy as np
import copy
import time
import json
# Define the function to minimize: Here is the maximum contact pressure
def fitness_function(xvar,case_name):
    
    import mesh_A1A2A3A4A5_2body_qin2 as mesh_A1A2A3A4A5 
    from templates import export_template
    import json
    if os.path.exists(str(pathlib.Path(__file__).parent.resolve())+'/'+case_name)==False:
        path = os.path.join(str(pathlib.Path(__file__).parent.resolve()), case_name)
        os.mkdir(path)  
    
    Fy=-2500
    
    
    #%% GEOMETRY
    l1=50
    l2=50
    
    h2=5
    h3=5
    h1=40
    h4=40
    h5=l2-(h3+h4)
    h0=l2-(h1+h2)
    lcontour=1*10
    lc=2
    
    jsonData=dict()
    jsonData['l2']=l2
    jsonData['lc']=lc
    jsonData['l1']=l1
    jsonData['h3']=h2
    jsonData['h2']=h2
    jsonData['h1']=h1
    jsonData['h0']=h0
    jsonData['h4']=h4
    jsonData['h5']=h5
    jsonData['lcontour']=lcontour
    
    with open(str(pathlib.Path(__file__).parent.resolve()) +'/'+case_name+'/parameters.json', "w") as outfile:
        json.dump(jsonData,outfile)  
    
    #%% MATERIAL PROPERTIES
    Eecc=200000
    Encc=80000
    Earti=200000
    Nuncc = 0.3;
    Nuecc = 0.3;
    
    #%% COMPUTE AND STORE RESULTS
    iteration_all=0
    foldersave='result_Encc'+str(int(Encc/1000))+'GPa'+'_Earti'+str(int(Earti/1000))
    if os.path.exists(str(pathlib.Path(__file__).parent.resolve())+'/'+case_name+'/'+foldersave)==False:
        
        path = os.path.join(str(pathlib.Path(__file__).parent.resolve())+'/'+case_name , foldersave)
        os.mkdir(path)    
    
    
       
    
    with open('parameters.json') as parameters:
        jsonData = json.load(parameters)
    jsonData={}    

    nameficheroi='plane_plane1_'
    
    ###### NEW DEVELOPMENT CODE 
    
    # ALL PATHS
    paths_dic = dict()
    
    paths_dic['main_root'] = str(pathlib.Path(__file__).parent.resolve())+'/'+case_name+'/'+foldersave
    paths_dic['aster_root'] = os.getenv('HOME')+'/salome_meca/appli_V2019.0.3_universal/salome shell -- as_run'
    paths_dic['salome_root'] = os.getenv('HOME')+'/salome_meca/appli_V2019.0.3_universal/salome'
    
    main_root = paths_dic['main_root']
    aster_root = paths_dic['aster_root']

    
    paths_dic['analysis_comm'] = main_root + '/analysis.comm'
    paths_dic['analyses_directory'] = main_root +'/'+ nameficheroi
    paths_dic['json_input_data'] = paths_dic['analyses_directory'] + '/input_params.json'
    
    
    # Find and Delete the existing ANALYSIS directory
    if os.path.exists(main_root+'/'+nameficheroi)==False:
        path = os.path.join(main_root, nameficheroi)
        os.mkdir(path)
    
    # Path to store data     
    required_dirs = ['UNV','RMED','MESS','EXPORT','COMM','VTK','JSON']
    
    for item in required_dirs:
        temp_path = paths_dic['analyses_directory'] + '/' + item
        paths_dic[item] = temp_path
        pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)
       
    
    namereper = {'name':'/'+case_name+'/'+foldersave+'/'+nameficheroi}      
    
    namereper_object = json.dumps(namereper, indent = 4)
        
        # Writing to sample.json
    with open('namereper.json', "w") as outfile:
            outfile.write(namereper_object)      


    plt.close('all')
    for num in range(1):
        
        print(num)
        Fyi=Fy/1
        iteration = copy.copy(num) #+ 1
        path_Eevo = main_root + '/E_evolution.txt'
        path_elems_criteria = main_root + '/elems_criteria.txt'

    
        path_export = paths_dic['EXPORT'] + f'/analysis_{iteration}.export'
        path_mess = paths_dic['MESS'] + f'/analysis_{iteration}.mess'
        path_comm = paths_dic['COMM'] + f'/analysis_{iteration}.comm'
        path_A1 = paths_dic['UNV'] + f'/analysis_{iteration}_A1.unv'
        path_A2 = paths_dic['UNV'] + f'/analysis_{iteration}_A2.unv'
        path_A3 = paths_dic['UNV'] + f'/analysis_{iteration}_A3.unv'
        path_A4 = paths_dic['UNV'] + f'/analysis_{iteration}_A4.unv'
        path_A5 = paths_dic['UNV'] + f'/analysis_{iteration}_A5.unv'
        path_rmed = paths_dic['RMED'] + f'/analysis_{iteration}.rmed'
        path_vtk = paths_dic['VTK'] + f'/analysis_{iteration}.vtk'
        elems_props_current = paths_dic['JSON'] + f'/dct_pgA2_{iteration}.json'
        elems_props_new = paths_dic['JSON'] + f'/dct_pgA2_{iteration + 1}.json'
        path_cp_values = paths_dic['RMED'] + f'/cp_values_{iteration}.json'
       
        # Input Data to be written
            
        temp_dic = {
            'iteration': iteration,
            'main_root': main_root,
            'path_A1': path_A1,
            'path_A2': path_A2,
            'path_A3': path_A3,
            'path_A4': path_A4,
            'path_A5': path_A5,
            'path_elems_criteria': path_elems_criteria,
            'path_rmed': path_rmed,
            'path_vtk': path_vtk,
            'elems_props_new': elems_props_new,
            'elems_props_current': elems_props_current,
            'path_cp_values': path_cp_values,
            'path_Eevo': path_Eevo,
            'Encc': Encc,
            'Eecc': Eecc,
            'Earti':Earti,
            'Eecc0': Eecc,
            'Ecc': Eecc,
            'Nuncc': Nuncc,
            'Nuecc': Nuecc,
            'Emin': Encc,
            'Emax': Eecc,
            'Fy': Fyi,
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,            
            'l2': l2,
            'lcontour':lcontour,
            'xvar':xvar,
            'repertoire':main_root + '/'+ nameficheroi,
            'fichier_folder':case_name+'/'+foldersave,
    
        }
    
        # Serializing json 
        json_object = json.dumps(temp_dic, indent = 4)
        
        # Writing to sample.json
        with open(paths_dic['json_input_data'], "w") as outfile:
            outfile.write(json_object)
        
    
        # Crate export files
        with open(path_export, 'w') as file:
            text = export_template(aster_root, path_comm, path_mess)
            file.write(text)
    
    
        with open('templates_mecano.comm', 'r') as file : 
       
        with open(path_comm, 'w') as file:
            file.write(filedata)
    
        
        if iteration_all==0:
            path_A1o=copy.copy(path_A1)
            path_A3o=copy.copy(path_A3)
            path_A5o=copy.copy(path_A5)
        
        mesh_A1A2A3A4A5.A1(path_A1,path_A1o,iteration_all,str(pathlib.Path(__file__).parent.resolve()) +'/'+case_name+'/parameters.json', disp_mesh = 0, mtype = 5, res = 7, localRefine = 1)
        mesh_A1A2A3A4A5.A3(path_A3,path_A3o,iteration_all,str(pathlib.Path(__file__).parent.resolve()) +'/'+case_name+'/parameters.json', disp_mesh = 0, mtype = 5, res = 15, localRefine = 1)
        mesh_A1A2A3A4A5.A5(path_A5,path_A5o,iteration_all,str(pathlib.Path(__file__).parent.resolve()) +'/'+case_name+'/parameters.json', disp_mesh = 0, mtype = 5, res = 15, localRefine = 1)
        
    
        # run the simulation
        run_file = path_export
    
        aster_run = Popen(aster_root+ " " + run_file, shell='True', executable="/bin/sh")
        aster_run.wait()
    
        x = list()
        cp = list()
        with open(path_cp_values,"r") as jsonfile:
            file = json.load(jsonfile)
    
        for key in file:
            if key!="energy_defor" and key!='s_mises_max'and key!='oi_seuil'and key!='necartilage':
                x.append(float(key))
                cp.append(file[key])
           
    plt.figure(1)
    plt.plot(x,cp,'b-',linewidth=1,label='contact pressure- GA')
    plt.plot(x,-Fy/l2*np.ones((len(cp),1)),'k--',linewidth=1,label='uniform contact pressure')
  

    plt.xlabel(r'Radius (mm)')
    plt.ylabel(r'Contact stress (MPa)')
    plt.title(r'Contact pressure vs radius')
    plt.xlim(0,l2+10)
    #ax.legend(prop=font)
    plt.legend()   
    plt.savefig(case_name+'/'+foldersave +'/pressure_Fy'+str(int(-Fy))+'_Ecartilage'+str(int(Encc/1000))+'GPa.eps')             

    maxcp=np.max(cp)  
    fitness_function.computations += 1    
    number_iter=dict()
    number_iter['number']=fitness_function.computations
    with open(str(pathlib.Path(__file__).parent.resolve()) +'/'+case_name+'/number_iter.json', "w") as outfile:
          json.dump(number_iter, outfile)                            
    return maxcp

    
# Function to create an individual with random genes
def create_individual(num_variables):
    return [random.choice([1, 0.4]) for _ in range(num_variables)]

# Function to create the initial population
def create_population(population_size, num_variables):
    return [create_individual(num_variables) for _ in range(population_size)]

# Function to evaluate the fitness of each individual in the population
def evaluate_population(population):
    return [fitness_function([*individual],case_name) for individual in population]

# Function to select parents for crossover using tournament selection
def tournament_selection(population, fitness_values, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_contestants = random.sample(list(enumerate(population)), tournament_size)
        winner = min(tournament_contestants, key=lambda x: fitness_values[x[0]])
        selected_parents.append(winner[1])
    return selected_parents

# Function to perform crossover between two parents to create two children
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.choice([1, 0.4])
    return mutated_individual

# Main genetic algorithm function
def genetic_algorithm(population_size, num_variables, generations, tournament_size, mutation_rate,case_name):
    population = create_population(population_size, num_variables)
    number_generationv=dict()
    for generation in range(generations):
        fitness_values = evaluate_population(population)
        best_individual_idx = fitness_values.index(min(fitness_values))
        best_individual = population[best_individual_idx]
        print(f"Generation {generation+1}: Best Individual: {best_individual}, Fitness: {fitness_values[best_individual_idx]}")

        parents = tournament_selection(population, fitness_values, tournament_size)
        next_generation = []
        
        while len(next_generation) < population_size:
            
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])
          
            
            print(len(next_generation))
        
        number_generationv['number']=generation
        number_generationv['x'+str(generation)]=best_individual
        number_generationv['maxcp'+str(generation)]=fitness_values[best_individual_idx]
         
        with open(str(pathlib.Path(__file__).parent.resolve())+'/' +case_name+'/number_generationv.json', "w") as outfile:
              json.dump(number_generationv, outfile)    

 
        population = next_generation
        time.sleep(600)

    return population

case_name='qin_rigidinterfaceaxis_adimensionee_genetic_80_E80'
population_size = 200# Population size of each generation 
num_variables = 80# design domains divides into 10x8 equal subdomains
generations = 50 # number of generation
tournament_size = 5# 
mutation_rate = 0.1
# Reset the computation counter before running the genetic algorithm
fitness_function.computations = 0
final_population = genetic_algorithm(population_size, num_variables, generations, tournament_size, mutation_rate,case_name)
best_individual = min(final_population, key=lambda x: fitness_function([*x],case_name))
print("\nFinal Result:")
print(f"Best Individual: {best_individual}, Fitness: {fitness_function([*best_individual],case_name)}")       

best_individualv=dict()
best_individualv['best_E']=best_individual
best_individualv['Fitness']=fitness_function(*best_individual,case_name)

with open(str(pathlib.Path(__file__).parent.resolve())+'/' +case_name+'/best_individualv.json', "w") as outfile:
      json.dump(best_individualv, outfile)  
