# -*- coding: utf-8 -*-
"""

# **Inteligência Artificial**
### Travelling salesman problem (TSP)

## **Introdução:**

O problema do caixeiro viajante é um desafio clássico da ciência da computação que consiste em encontrar o menor caminho possível para que um caixeiro viaje por todas as cidades de um conjunto dado, passando por cada cidade uma única vez e retornando à cidade de origem. Embora existam soluções exatas para o problema em pequenas instâncias, conforme o número de cidades aumenta, encontrar a solução ótima se torna impraticável computacionalmente. Nesse contexto, algoritmos genéticos surgem como uma alternativa eficiente para a solução aproximada do problema. Através da seleção e combinação de soluções parciais, esses algoritmos podem encontrar soluções de alta qualidade em tempo razoável. Este trabalho tem como objetivo aplicar algoritmos genéticos para resolver instâncias do problema do caixeiro viajante e avaliar sua efetividade em termos de qualidade das soluções encontradas e tempo de execução.

### **Aplicações**

Algumas das principais aplicações do TSP são:

* Planejamento de rotas de transporte: o TSP é usado para otimizar as rotas de transporte de bens e mercadorias, reduzindo o tempo e os custos envolvidos na entrega.

* Design de circuitos eletrônicos: o TSP é usado para otimizar o layout de circuitos eletrônicos, reduzindo o tempo e o espaço necessários para conectar todos os componentes do circuito.

* Sequenciamento de DNA: o TSP é usado para otimizar o processo de sequenciamento do DNA, reduzindo o tempo e o custo envolvidos na determinação da sequência de nucleotídeos.

* Roteamento de veículos: o TSP é usado para otimizar o roteamento de veículos em diversas aplicações, incluindo coleta de lixo, transporte público e entrega de encomendas.

Em resumo, o TSP tem uma ampla variedade de aplicações práticas em diversos campos, ajudando a resolver problemas complexos de otimização e reduzindo os custos e o tempo envolvidos em processos logísticos, de transporte e de produção.

## **Instancias a serem resolvidadas**:

#### **Berlin52**

Um das  instância  do problema TSP que iremos resolver é BERLIN52 na qual que consiste em encontrar o menor caminho possível que um caixeiro viaje por todas as 52 localidades de Berlim, Alemanha. Essa instância é frequentemente usada como um benchmark para avaliar o desempenho de algoritmos que resolvem o TSP.  O único
critério de otimização é a distância para completar o
jornada. A solução ótima para este problema é conhecida, 7542  metros.
"""

import matplotlib.pyplot as plt
import numpy as np

data_berlin= { 1:(565, 575), 2:(25, 185), 3:(345, 750), 4:(945, 685), 5:(845, 655), 6: (880, 660), 7: (25, 230), 8: (525, 1000), 9: (580, 1175), 10: (650, 1130),
              11: (1605, 620), 12: (1220, 580),13: (1465, 200), 14: (1530, 5), 15: (845, 680), 16: (725, 370), 17: (145, 665), 18: (415, 635), 19: (510, 875), 20: (560, 365),
              21: (300, 465),  22: (520, 585), 23: (480, 415), 24: (835, 625), 25: (975, 580), 26: (1215, 245), 27: (1320, 315), 28: (1250, 400), 29: (660, 180), 30: (410, 250), 
              31: (420, 555),  32: (575, 665), 33: (1150, 1160), 34: (700, 580), 35: (685, 595), 36: (685, 610), 37: (770, 610), 38: (795, 645), 39: (720, 635), 40: (760, 650), 
              41: (475, 960),  42: (95, 260),  43: (875, 920), 44: (700, 500), 45: (555, 815), 46: (830, 485), 47: (1170, 65), 48: (830, 610), 49: (605, 625), 50: (595, 360), 
              51: (1340, 725), 52: (1740, 245)}

x,y= zip(*data_berlin.values())



plt.scatter(x,y,label='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Berlin52')
plt.show()

"""#### **pr76**

Outra instância que  resolveremos será pt76 para 76 locais seguindo o mesmo critério. A solução ótima para este problema também é conhecida, 108159  metros.
"""

data_pr76 = {1: (3600, 2300),2: (3100, 3300),3: (4700, 5750),4: (5400, 5750),5: (5608, 7103),6: (4493, 7102), 7: (3600, 6950),8: (3100, 7250),9: (4700, 8450),10: (5400, 8450),
            11: (5610, 10053),12: (4492, 10052),13: (3600, 10800),14: (3100, 10950),15: (4700, 11650),16: (5400, 11650),17: (6650, 10800),18: (7300, 10950),19: (7300, 7250),20: (6650, 6950),
            21: (7300, 3300),22: (6650, 2300),23: (5400, 1600),24: (8350, 2300),25: (7850, 3300),26: (9450, 5750),27: (10150, 5750),28: (10358, 7103),29: (9243, 7102),30: (8350, 6950),
            31: (7850, 7250),32: (9450, 8450),33: (10150, 8450),34: (10360, 10053),35: (9242, 10052),36: (8350, 10800),37: (7850, 10950),38: (9450, 11650),39: (10150, 11650),40: (11400, 10800),
            41: (12050, 10950),42: (12050, 7250),43: (11400, 6950),44: (12050, 3300),45: (11400, 2300),46: (10150, 1600),47: (13100, 2300),48: (12600, 3300),49: (14200, 5750),50: (14900, 5750),
            51: (15108, 7103),52: (13993, 7102),53: (13100, 6950),54: (12600, 7250),55: (14200, 8450),56: (14900, 8450),57: (15110, 10053),58: (13992, 10052),59: (13100, 10800),60: (12600, 10950),
            61: (14200, 11650), 62: (14900,11650), 63: (16150, 10800), 64: (16800, 10950), 65: (16800, 7250), 66: (16150, 6950), 67: (16800, 3300), 68: (16150, 2300), 69: (14900, 1600), 70: (19800, 800), 
            71: (19800, 10000),  72: (19800, 11900), 73: (19800, 12200), 74: (200, 12200), 75: (200, 1100), 76: (200, 800)}


x,y= zip(*data_pr76.values())

plt.scatter(x,y,label='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('pr76')
plt.show()

"""#### **st70**

Ultimo teste será o st70, na qual  são 70 locais diferentes.Como as outras duas aa solução ótima é conhecida 675 e o critério é o mesmo.
"""

data_st70 = {1: (64, 96), 2: (80, 39), 3: (69, 23), 4: (72, 42), 5: (48, 67), 6: (58, 43), 7: (81, 34), 8: (79, 17), 9: (30, 23), 10: (42, 67), 
             11: (7, 76), 12: (29, 51), 13: (78, 92), 14: (64, 8), 15: (95, 57), 16: (57, 91), 17: (40, 35), 18: (68, 40), 19: (92, 34), 20: (62, 1), 
             21: (28, 43), 22: (76, 73), 23: (67, 88), 24: (93, 54), 25: (6, 8), 26: (87, 18), 27: (30, 9), 28: (77, 13), 29: (78, 94), 30: (55, 3),
             31: (82, 88), 32: (73, 28), 33: (20, 55), 34: (27, 43), 35: (95, 86), 36: (67, 99), 37: (48, 83), 38: (75, 81), 39: (8, 19), 40: (20, 18),
             41: (54, 38), 42: (63, 36), 43: (44, 33), 44: (52, 18), 45: (12, 13), 46: (25, 5), 47: (58, 85), 48: (5, 67), 49: (90, 9), 50: (41, 76),
             51: (25, 76), 52: (37, 64), 53: (56, 63), 54: (10, 55), 55: (98, 7), 56: (16, 74), 57: (89, 60), 58: (48, 82), 59: (81, 76), 60: (29, 60), 
             61: (17, 22), 62: (5, 45), 63: (79, 70), 64: (9, 100), 65: (17, 82), 66: (74, 67), 67: (10, 68), 68: (48, 19), 69: (83, 86), 70: (84, 94)}

x,y= zip(*data_st70.values())



plt.scatter(x,y,label='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('st70')
plt.show()

"""### **Complexidade**

Para entendermos a complexidade computacional desse problema,vamos dar uma olhada no seguinte exemplo, imagine que temos que estamos na cidade A e partir dela queremos visitar as cidades B,C e D, assim como no problema do caixeiro viajante podemos visitar as cidades em qualquer ordem e temos que voltar a cidade que começamos.

Podemos calcular o numero de rotas possivel na seguinte maneira:

$R(n)=(n-1)!$

Logo para 4 cidades concluiriamos que:

$R(n)=3!=6$

Percebemos que existem 6 rotas possiveis para exemplo. Agora olhando para as instâncias que queremos resolver, assim teremos que:

$R(52)=51!= 1551118753287382280224243016469303211063259720016986112000000000000$

$R(70)=70!= 11978571669969891796072783721689098736458938142546425857555362864628009582789845319680000000000000000$

$R(76)=76!=1885494701666050254987932260861146558230394535379329335672487982961844043495537923117729972224000000000000000000$

Se colocarmos  um computador para realizar qualquer uma dessas tarefas, tempo necessário para completa-la  seria de milhares de anos. Por termos problema com complexidade de tempo de $O(n!)$, significa  número de instruções executadas cresce muito rapidamente para um pequeno crescimento do número de itens processados, tornando impraticavel o computador executar a tarefa em tempo viavel.

O problema do caixeiro viajante é um  clássico combinatório NP-completo, ou seja podemos verificar a solução em tempo polinomial. É o que faremos utilizando o  Algoritmo Genetico.

## **Algoritmo**

A seguir, apresenta-se a implementação de um algoritmo genético, o qual recebe os seguintes parâmetros: população inicial (population), função de avaliação de soluções (fn_fitness), conjunto de valores possíveis para cada posição da solução (gene_pool), função de critério de parada por qualidade da solução (fn_thres), número máximo de gerações (ngen) e probabilidade de mutação (pmut). Ao final da execução do algoritmo, é retornado o melhor indivíduo (solução) da última geração de indivíduos.
"""

import random as random
import numpy as np
import bisect
import math

def genetic_algorithm(population, fn_fitness, gene_pool, select, crossover, mutate, fn_thres=None, ngen=1000, pmut=0.1):
    save=[]
    # for each generation
    for i in range(ngen):

        # create a new population
        new_population = []

        # repeat to create len(population) individuals
        for i in range(len(population)):
          
          # select the parents
          p1, p2 = select(2, population, fn_fitness, True)

          # recombine the parents, thus producing the child
          child = crossover(p1, p2)

          # mutate the child
          child = mutate(child, gene_pool, pmut)

          # add the child to the new population
          new_population.append(child)

        # move to the new population
        population = new_population

        # Store the individual with highest  fitnesss
        aux=max(population, key=fn_fitness)        
        store=1/fn_fitness(aux)
        save.append(store)

        # check if one of the individuals achieved a fitness of fn_thres; if so, return it
        fittest_individual = fitness_threshold(fn_fitness, fn_thres, population)
        if fittest_individual:
            return fittest_individual

    # plot distance x generation graph
    plt.plot(range(ngen), save)
    plt.ylabel('Distância')
    plt.xlabel('Geração')
    plt.show()
  
    # return the individual with highest fitness
    return max(population, key=fn_fitness)
  
# get the best individual of the received population and return it if its 
# fitness is higher than the specified threshold fn_thres
def fitness_threshold(fn_fitness, fn_thres, population):
    if not fn_thres:
        return None

    fittest_individual = max(population, key=fn_fitness)
    if fn_fitness(fittest_individual) >= fn_thres:
        return fittest_individual

    return None

# ----------------- Usado para gerar  exemplos--------------------

# evaluation class; 
# since that a solution needs to be evaluated with respect to the problem instance 
# in consideration, we created this class to store the problem instance and to 
# allow the evaluation to be performed without having the problem instance at hand
class EvaluateGC:
    # during initialization, store the problem instance
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
    
    # compute the value of the received solution
    def __call__(self, solution):
        return sum(solution[n1] != solution[n2] for (n1, n2) in self.problem_instance.values())

"""#### **Indivíduo**

Representa de uma possivel solução para o problema em questão, para entendermos isso vamos olhar o exemplo que apresentamos anteriormente,na qual  estavamos cidade A e partir dela queremos visitar as cidades B,C e D, e voltar para nossa casa na cidade A

Um possivel solução  esse caminho seria `A->B->D->C->A`:

Essa  routa poderimos codificar como  sendo um array que começaria no cidade A, passaria pelo cidade B, depois pelo cidade D, até voltar cidade A, seguindo a ordem estabelecida pelo vetor.
"""

Rota=['A', 'B', 'D' ,'C', 'A']

"""Em nosso caso será um arranjo formado com os `id` será a sequência das cidades a ser visitada, abaixo temos um exemplo possivel solução para instância berlim52:


#### **Vizinhança**

Para fazer a análise da vizinhança vamos  pegar o  Indivíduo  que apresentamos anteriormente, rota `A->B->D->C->A` e vemos outros soluções que poderiam ser obtidas a partir desta rota inicial fazendo pequenas alterações. Aqui abaixo temos alguns exemplos

Logo podemos escrever a vizinhança como  sendo:

$vizinhança= (n-1)^2$

Onde $n$ é tamanho do individuo, então a vizinhança das nossas instâncias podemos escrever como  sendo:

$Berlin52= 51^2=2601$

$st70=69^2=4.761$

$pr76=75^2=5.625$

Estamos considerando que indepente do ponto que começamos devemos voltar a ele no final.

#### **População**

Um dos parâmetros que tem  grande influencia sobre o desempenho e a eficiência dos algoritmos  é a População. Se for muito pequena não garante uma alta cobertura do espaço de busca e população maiores exigiram um poder computacional e mais para execução do algoritmo.

Para nosso projeto faremos a inicialização da população incial de forma randômica,ou seja daremos valores aleatórios a cada gene dos cromossomos. Também planejamos para o tamanho da nossa população como sendo 200, para garantir qualidade na busca.
"""

def init_population(pop_number, gene_pool, state_length):
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = random.sample(gene_pool, g)
        population.append(new_individual)

    return population

# usando a titulo de exemplo
problem_instance = {
    '1-5': [0, 4],
    '1-2': [0, 1],
    '1-6': [0, 5],
    '2-7': [1, 6],
    '2-3': [1, 2],
    '3-8' : [2, 7],
    '3-4': [2, 3],
    '4-5' : [3, 4],
    '4-9': [3, 8],
    '5-10': [4, 9],
    '6-9' : [5, 8],
    '6-8': [5, 7],
    '7-10': [6, 9],
    '7-9': [6, 8],
    '8-10': [7, 9]
}

population_size = 10
individual_length = 10
possible_values = list(range(population_size)) # cuidar esse parametro aqui. Impacta na mutação caso exista divergência.

# create initial population from problem instance
population = init_population(population_size,possible_values, individual_length)
# create an instance of the evaluation class for the considered problem instance
fn_fitness = EvaluateGC(problem_instance)

print('Resulting population: %s' % population)

"""#### **Função de Adaptação (Fitness)**

Responsável por avaliar cada possivel solução em virtude quão bem um dado candidato à solução é capaz de resolver o problema. Em nosso caso os mais aptos serão aqueles que retornarem a menor valores  para custo total da viagem.

Para modelarmos nossa função devemos considerar as restrições que estão impostas sobre o nosso problema. Como o custo da aresta entre dois nós A e B é da pela distância Euclidiana entre esses dois nós:


$d(A,B)=\sqrt[]{(x_B-x_A)^2+(y_B-y_A)^2}$

Como nossa meta é calcular a distância total a ser percorrida, levando em conta que partimos do ponto inicial($1$), logo teremos que passar por todas $n$ cidades e retomar a cidade que começamos. Podemos então escrever isso da seguinte forma:

$Dtotal=d(1,2)+d(2,3)+d(3,4)+...+d(n-1,n)+d(n,1)$

A equação  representada a cima é a queremos otimizar a minimizando para encontrarmos nosso solução ótima, como nosso  codigo AG  trabalho maximando  o valor de fitness,com isso iremos tratar o fitness como  sendo inversamente proporcional ao tamanho da rota
"""

from IPython.utils.text import format_screen
# evaluation class 
class EvaluateTSP:
    # during initialization, store the problem instance
    def __init__(self, n):
          self.n = n
    # compute the value of the received solution
    def __call__(self, individuo):  
      distance=0
      for i in range(0, len(individuo)):
        gene=self.n.get(individuo[i])
        if(i < len(individuo) - 1): 
          nextGene = self.n.get(individuo[i+1])
          xDis = abs(gene[0] - nextGene[0])
          yDis = abs(gene[1] - nextGene[1])
        else:
          nextGene = self.n.get(individuo[0])
          xDis = abs(gene[0] - nextGene[0])
          yDis = abs(gene[1] - nextGene[1])
        
        penalty = 0
        for j in range(0, i):
          if individuo[i] == individuo[j]:
            penalty +=  100000

        distance = distance + np.sqrt((xDis ** 2) + (yDis ** 2)) + penalty

      if distance == 0:
        distance = 1
      return 1/distance

"""A equação  representada a cima é a queremos otimizar a minimizando para encontrarmos nosso solução ótima.

#### **Operadores genéticos utilizados na solução**

##### **Seleção**

Para o operador seleção, estaremos utilizando os seguintes operadores:

*   Torneio

O operador de seleção por torneio funciona selecionando aleatoriamente um subconjunto da população (chamado de "pool de torneio") e escolhendo o indivíduo com o melhor desempenho nesse subconjunto. O tamanho do pool de torneio é um parâmetro que pode ser ajustado, geralmente entre 2 e 10 indivíduos. Esse processo é repetido várias vezes para selecionar os indivíduos que serão usados para reprodução.
"""

# genetic operator for selection of individuals; 
# this function implements roulette wheel selection, where individuals with 
# higher fitness are selected with higher probability
def tourneySelection(parents, population, fn_fitness, stopLog=False):
    selected_parents = []
    for i in range(parents):
        tournament = random.sample(population, 10) #definir tamanho do torneio
        if(not stopLog):
          print("tournament %s:" % tournament)

        fitnesses = list(map(fn_fitness, tournament))
        if(not stopLog):
          print("fitnesses %s" % fitnesses)

        winner = tournament[fitnesses.index(max(fitnesses))]
        selected_parents.append(winner)
    
    return tuple(selected_parents) # com uma população pequena, isso pode resultar no mesmo individuo para p1 e p2

#sample
p1,p2 = tourneySelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)

"""*   Roleta

O operador de seleção por roleta funciona de forma probabilística, onde a probabilidade de um indivíduo ser selecionado para reprodução é proporcional à sua aptidão (fitness) em relação à população inteira. Esse processo é semelhante a uma roleta, onde cada indivíduo recebe uma fatia proporcional ao seu fitness, e uma "roleta" é girada para escolher qual indivíduo será selecionado. Esse processo é repetido várias vezes para selecionar os indivíduos que serão usados para reprodução.
"""

# genetic operator for selection of individuals; 
# this function implements roulette wheel selection, where individuals with 
# higher fitness are selected with higher probability
def rouletteSelection(parents, population, fn_fitness, stopLog=False):
    fitnesses = list(map(fn_fitness, population))
    if(not stopLog):
      print("population %s" % population)
      print("fitnesses %s" % fitnesses)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(parents)]

# return a single sample from seq; the probability of a sample being returned
# is proportional to its weight
def weighted_sampler(seq, weights):
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

#sample
p1,p2 = rouletteSelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)

"""##### **Crossover**

Para o operador de crossover, estaremos utilizando os seguintes operadores:

*   Blend

O blend crossover é um método que combina os genes de ambos os pais de forma mais suave, levando em consideração a diferença entre seus valores. Nesse método, um ponto de corte é escolhido aleatoriamente como no single point crossover, mas em vez de simplesmente trocar os segmentos antes e depois desse ponto, o blend crossover calcula novos valores para cada gene dos descendentes com base nos valores dos pais e em um parâmetro de mistura.

"""

def blendCrossover(p1, p2, alpha=0.5):
    # Create a list to store the child's values
    child = []
    
    # For each value in the parents, calculate a new value for the child using the Blend crossover formula
    for i in range(len(p1)):
        # Calculate the minimum and maximum possible values for the child's ith value
        min_value = min(p1[i], p2[i])
        max_value = max(p1[i], p2[i])
        
        # Calculate the range of values for the child's ith value using the alpha blend factor
        range_value = (max_value - min_value) * alpha
        
        # Calculate the lower and upper bounds for the child's ith value
        lower_bound = min_value - range_value
        upper_bound = max_value + range_value
        
        # Generate a random value within the bounds for the child's ith value and append it to the child's list of values
        child.append(math.floor(random.uniform(lower_bound, upper_bound)))
    
    return child


#sample
print('Sample using roulette')
p1,p2 = rouletteSelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % blendCrossover(p1,p2))

print('-------------------')

print('Sample using tourney')
p1,p2 = tourneySelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % blendCrossover(p1,p2))

"""*   Double Point

No operador de double point é combinado dois cromossomos parentais para criar dois novos cromossomos filhos. Ele funciona selecionando dois pontos aleatórios em cada cromossomo parental e trocando os segmentos entre esses pontos para criar os cromossomos filhos.

"""

def doublePointCrossover(p1, p2):
    # Choose two random crossover points
    point_1 = random.randint(0, len(p1) - 1)
    point_2 = random.randint(0, len(p1) - 1)

    # Swap the segments between the crossover points
    if point_1 > point_2:
        point_1, point_2 = point_2, point_1
    child_1 = p1[:point_1] + p2[point_1:point_2] + p1[point_2:]
    child_2 = p2[:point_1] + p1[point_1:point_2] + p2[point_2:]

    f1 = fn_fitness(child_1)
    f2 = fn_fitness(child_2)

    if f1 > f2:
      return child_1
    else:
      return child_2

"""*   Single Point

O single point é um método de crossover em que um ponto de corte é escolhido aleatoriamente na cadeia de genes dos pais. Os segmentos de genes antes e depois do ponto de corte são então trocados entre os pais, produzindo dois descendentes.

"""

def singlePointCrossover(p1, p2):
    # Choose a random crossover point between 0 and the length of the individuals
    crossover_point = random.randint(0, len(p1) - 1)
    
    # Create the child by taking the first part of p1 and the second part of p2
    child = p1[:crossover_point] + p2[crossover_point:]
    
    return child

#sample
print('Sample using roulette')
p1,p2 = rouletteSelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % singlePointCrossover(p1,p2))

print('-------------------')

print('Sample using tourney')
p1,p2 = tourneySelection(2, population, fn_fitness)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % singlePointCrossover(p1,p2))

"""##### **Mutação**

Para o operador de mutação, estaremos utilizando os seguintes operadores:

* Uniform

O algoritmo de mutação uniforme é uma técnica simples de mutação usada em algoritmos genéticos. Ele consiste em selecionar aleatoriamente um gene de um indivíduo e mutá-lo de acordo com uma determinada probabilidade de mutação. Essa probabilidade de mutação é um valor fixo que determina a chance do gene ser ou não mutado.
"""

def uniformMutation(x, gene_pool, pmut):
    
    # if random >= pmut, then no mutation is performed
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n) # gene to be mutated
    r = random.randrange(0, g) # new value of the selected gene

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c+1:]

#sample
print('Sample using ROULETTE with BLEND crossover')
p1,p2 = rouletteSelection(2, population, fn_fitness)
crossover = blendCrossover(p1,p2)
mutatedChild = uniformMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

print('Sample using TOURNEY with BLEND crossover')
p1,p2 = tourneySelection(2, population, fn_fitness)
crossover = blendCrossover(p1,p2)
mutatedChild = uniformMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

#sample 
print('Sample using ROULETTE with SINGLE-POINT crossover')
p1,p2 = rouletteSelection(2, population, fn_fitness)
crossover = singlePointCrossover(p1,p2)
mutatedChild = uniformMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

print('Sample using TOURNEY with SINGLE-POINT crossover')
p1,p2 = tourneySelection(2, population, fn_fitness)
crossover = singlePointCrossover(p1,p2)
mutatedChild = uniformMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

"""*   Boundary

É um método da mutação que limita os novos valores dos genes dentro de um intervalo definido, para evitar que a mutação leve a valores inválidos ou impraticáveis. Esse método é especialmente útil em problemas em que as soluções devem satisfazer certas restrições ou limitações. A mutação boundary pode ser implementada de diferentes maneiras, como simplesmente truncar os novos valores para o intervalo permitido, ou ajustar o tamanho da mutação de forma a evitar que os novos valores ultrapassem os limites.
"""

def boundaryMutation(child, gene_pool, mutation_probability):
    # Create a list to store the mutated child's values
    mutated_child = []
    
    # For each value in the child, determine whether to mutate it or not based on the mutation probability
    for i in range(len(child)):
        if random.random() < mutation_probability:
            # If the gene is to be mutated, randomly select a new value from the gene pool
            new_value = random.choice(gene_pool)
        else:
            # If the gene is not to be mutated, use the original value from the child
            new_value = child[i]
        
        # Check if the new value is within the allowed bounds for this gene, and if not, set it to the nearest allowed value
        if new_value < min(gene_pool):
            new_value = min(gene_pool)
        elif new_value > max(gene_pool):
            new_value = max(gene_pool)
        
        # Append the new value to the mutated child's list of values
        mutated_child.append(new_value)
    
    return mutated_child

#sample
print('Sample using ROULETTE with BLEND crossover')
p1,p2 = rouletteSelection(2, population, fn_fitness)
crossover = blendCrossover(p1,p2)
mutatedChild = boundaryMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

print('Sample using TOURNEY with BLEND crossover')
p1,p2 = tourneySelection(2, population, fn_fitness)
crossover = blendCrossover(p1,p2)
mutatedChild = boundaryMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

#sample 
print('Sample using ROULETTE with SINGLE-POINT crossover')
p1,p2 = rouletteSelection(2, population, fn_fitness)
crossover = singlePointCrossover(p1,p2)
mutatedChild = boundaryMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

print('-------------------')
print('-------------------')

print('Sample using TOURNEY with SINGLE-POINT crossover')
p1,p2 = tourneySelection(2, population, fn_fitness)
crossover = singlePointCrossover(p1,p2)
mutatedChild = boundaryMutation(crossover, possible_values, 0.15)
print('p1 %s:' % p1)
print('p2 %s:' % p2)
print('crossover %s:' % crossover)
print('mutation %s:' % mutatedChild)

"""##### **Estado da Arte**

  O algoritmo utilizado para o estado da arte foi o Ant Colony Optimization (ACO). A ideia por trás do ACO é inspirada no comportamento das formigas em encontrar o caminho mais curto entre uma colônia de formigas e uma fonte de alimento. Elas usam um processo de comunicação baseado em feromônios para marcar os caminhos percorridos, orientando assim outras formigas a seguirem o mesmo caminho.

  No ACO, cada cromossomo (solução) é visto como uma formiga em busca de uma solução ótima para um problema. O algoritmo usa uma série de feromônios para guiar a busca pelas soluções mais promissoras. Cada elemento da matriz representa a quantidade de feromônio depositado por uma formiga em um determinado local do espaço de busca.

  À medida que o algoritmo evolui, os cromossomos colocam feromônios em locais no espaço de busca que levam a melhores soluções. Dessa forma, as soluções mais promissoras acabam sendo reforçadas e se tornam mais atrativas para outras formigas (cromossomos) durante a evolução.

  ACO é um operador genético eficiente para problemas de otimização combinatória onde a solução consiste em uma sequência de valores discretos. Ele tem sido usado em muitas aplicações, como roteamento de veículos, programação de produção e programação.


"""

!pip install acopy==0.7.0

!pip install tsplib95

import acopy
import numpy
import tsplib95
import matplotlib.pyplot as plt
import pandas as pd

# Convert the dictionary to a TSPLIB file
with open("berlin52.tsp", "w") as f:
    f.write("NAME: berlin52\n")
    f.write("TYPE: TSP\n")
    f.write(f"COMMENT: {len(data_berlin)} cities in Berlin\n")
    f.write("DIMENSION: {}\n".format(len(data_berlin)))
    f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")
    for i, coord in data_berlin.items():
        f.write("{} {} {}\n".format(i, coord[0], coord[1]))
    f.write("EOF\n")

# Load the TSPLIB file
problem = tsplib95.load("berlin52.tsp")
G = problem.get_graph()

solver = acopy.Solver(rho=.003, q=0.2)
colony = acopy.Colony(alpha=1, beta=3)
tour = solver.solve(G, colony, limit=450)

print(tour.cost)
print(tour.get_id())
print(tour.nodes)
print(tour.path)

X_new = []
Y_new = []
for i in tour.path:
    X_new.append(i[0]-1)
    Y_new.append(i[1]-1)

plt.scatter(X_new,Y_new, c = "orange")
plt.plot(X_new,Y_new, c = "purple")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

"""#### **Criterio de Parada**

os critérios de parada são:
*  `generations`: Quando o AG atinge um determinado número de gerações ou verificações;
* `threshold`: Quando o valor ótimo da função objetivo for atingido (caso este seja conhecido);
* Quando não houver melhoramento razoável no melhor cromossomo por um
determinado número de gerações

## **Avaliação de Resultados**
"""

population_size=100
generations=100
mutation=0.1

"""### **Berlin52**"""

entrada=data_berlin
individual_length = 52 #Indiviual_length é o tamanho do Individuo
possible_values = sorted(data_berlin) # cuidar esse parametro aqui. Impacta na mutação caso exista divergência. #possible_values tem que  ser valores possiveis nosso caso cidades(X,Y)


# create initial population from problem instance
population = init_population(population_size, possible_values, individual_length)
# create an instance of the evaluation class for the considered problem instance
fn_fitness = EvaluateTSP(entrada)
threshold=1/7542

"""##### **Tourney + doublePointCrossover + Uniform**"""

# Tourney + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + doublePointCrossover + Uniform**"""

# Roulette + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Uniform**"""

#  Tourney + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, Single Point, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Uniform**"""

# Roulette + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Double Point Crossover + Boundary**"""

# Tourney + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Double Point Crossover + Boundary**"""

# Roulette + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Boundary**"""

# Tourney + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Boundary**"""

# Roulette + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Estado da Arte**"""

# Load the TSPLIB file
problem = tsplib95.load("berlin52.tsp")
G = problem.get_graph()

solver = acopy.Solver(rho=.003, q=0.2)
colony = acopy.Colony(alpha=1, beta=3)
tour = solver.solve(G, colony, limit=450)

print(tour.cost)
print(tour.get_id())
print(tour.nodes)
print(tour.path)

"""### **pr76**


"""

# Exemplo de execução
entrada=  data_pr76
population_size = 100
individual_length = 76 #Indiviual_length é o tamanho do Individuo
possible_values = sorted(data_pr76) # cuidar esse parametro aqui. Impacta na mutação caso exista divergência. #possible_values tem que  ser valores possiveis nosso caso cidades(X,Y)

# create initial population from problem instance
population = init_population(population_size, possible_values, individual_length)
# create an instance of the evaluation class for the considered problem instance
fn_fitness = EvaluateTSP(entrada)
threshold=1/108159

"""##### **Tourney + doublePointCrossover + Uniform**"""

# Tourney + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=500, pmut=mutation)

print("Solution (tourney, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + doublePointCrossover + Uniform**"""

# Roulette + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Uniform**"""

#  Tourney + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, Single Point, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Uniform**"""

# Roulette + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Double Point Crossover + Boundary**"""

# Tourney + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Double Point Crossover + Boundary**"""

# Roulette + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Boundary**"""

# Tourney + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Boundary**"""

# Roulette + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Estado da Arte**"""

# Convert the dictionary to a TSPLIB file
with open("pr76.tsp", "w") as f:
    f.write("NAME: pr76\n")
    f.write("TYPE: TSP\n")
    f.write(f"COMMENT: {len(data_pr76)} cities in Berlin\n")
    f.write("DIMENSION: {}\n".format(len(data_pr76)))
    f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")
    for i, coord in data_pr76.items():
        f.write("{} {} {}\n".format(i, coord[0], coord[1]))
    f.write("EOF\n")

# Load the TSPLIB file
problem = tsplib95.load("pr76.tsp")
G = problem.get_graph()

solver = acopy.Solver(rho=.003, q=0.2)
colony = acopy.Colony(alpha=1, beta=3)
tour = solver.solve(G, colony, limit=450)

print(tour.cost)
print(tour.get_id())
print(tour.nodes)
print(tour.path)

"""### **st70**"""

# Exemplo de execução
entrada=  data_st70
population_size = 100
individual_length = 70 #Indiviual_length é o tamanho do Individuo
possible_values = sorted(data_st70) # cuidar esse parametro aqui. Impacta na mutação caso exista divergência. #possible_values tem que  ser valores possiveis nosso caso cidades(X,Y)

# create initial population from problem instance
population = init_population(population_size, possible_values, individual_length)
# create an instance of the evaluation class for the considered problem instance
fn_fitness = EvaluateTSP(entrada)
threshold=1/675

"""##### **Tourney + doublePointCrossover + Uniform**"""

# Tourney + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=500, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + doublePointCrossover + Uniform**"""

# Roulette + doublePointCrossover + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, doublePointCrossover, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Uniform**"""

#  Tourney + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, Single Point, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Uniform**"""

# Roulette + Single Point + Uniform
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, uniformMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, uniform): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Double Point Crossover + Boundary**"""

# Tourney + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Double Point Crossover + Boundary**"""

# Roulette + Double Point Crossover + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, doublePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Tourney + Single Point + Boundary**"""

# Tourney + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, tourneySelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (tourney, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Roulette + Single Point + Boundary**"""

# Roulette + Single Point + Boundary
solution = genetic_algorithm(population, fn_fitness, possible_values, rouletteSelection, singlePointCrossover, boundaryMutation, fn_thres=threshold, ngen=generations, pmut=mutation)

print('\n-------------------\n')

print("Solution (roulette, blend, boundary): %s" % solution)
print(1/fn_fitness(solution))

"""##### **Estado da Arte**"""

# Convert the dictionary to a TSPLIB file
with open("st70.tsp", "w") as f:
    f.write("NAME: st70\n")
    f.write("TYPE: TSP\n")
    f.write(f"COMMENT: {len(data_st70)} cities in Berlin\n")
    f.write("DIMENSION: {}\n".format(len(data_st70)))
    f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")
    for i, coord in data_st70.items():
        f.write("{} {} {}\n".format(i, coord[0], coord[1]))
    f.write("EOF\n")

# Load the TSPLIB file
problem = tsplib95.load("st70.tsp")
G = problem.get_graph()

solver = acopy.Solver(rho=.003, q=0.2)
colony = acopy.Colony(alpha=1, beta=3)
tour = solver.solve(G, colony, limit=450)

print(tour.cost)
print(tour.get_id())
print(tour.nodes)
print(tour.path)