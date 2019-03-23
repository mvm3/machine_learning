import pandas as pd
import numpy as np
from random import *
from collections import *
import copy
import time

class Pattern:
    def __init__(self, lista, fit):
        self.lista = lista
        self.fit = fit

    def __repr__(self):
        return repr((self.lista, self.fit))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self,other):
        return self.lista == other.lista

def WRAcc(linha_mtz):
    if not linha_mtz:
        return 0.0
    b = np.full((len(df)),True)
    for i in linha_mtz:
        a = matrz[i]
        auxL = [x and y for (x, y) in zip(a, b)]
        b = auxL
    TP = 0
    FP = 0
    for i in range(D):
        if i<=Dmais and b[i] == 1:
            TP = TP + 1
        elif i>Dmais and b[i] ==1:
            FP = FP + 1
    if(TP+FP == 0):
        aux = 0
    else:
        aux = ((TP + FP)/D)
        aux2 = (TP/(TP + FP)) - (Dmais/D)
        aux= aux*aux2*4
    return aux

def INICIALIZAR_D1():
    P0 = []
    for i in range(2*len(columns)):# GERA P
        P0.append(Pattern([i],0))
        P0[i].fit = WRAcc(P0[i].lista)
    return P0

def INICIALIZAR_ALEATORIO1_D_Pk(tamanhoPopulacao, Pk):
    total = 0
    for i in range(len(Pk)):
        total = total + len(Pk[i].lista)
    n_dimensoes = total//(len(Pk))

    if(n_dimensoes<2):
        n_dimensoes = 2
    
    P0 = []

    for i in range(9*tamanhoPopulacao//10):
        Vaux = set()

        while(len(Vaux) < n_dimensoes):
            Vaux.add(randint(0,(tamanhoPopulacao-1)))

        P0.append(Pattern(list(Vaux),0))
        P0[i].fit = WRAcc(P0[i].lista)
        
    
    itensPk = set()
    for i in range(len(Pk)):
        for j in range(len(Pk[i].lista)):
            itensPk.add(Pk[i].lista[j])

    itensPk = list(itensPk)
    shuffle(itensPk)

    for i in range(tamanhoPopulacao-(9*tamanhoPopulacao//10)):
        Vaux = set()
        aux = []
        while(len(Vaux) < n_dimensoes):
            Vaux.add(itensPk[randint(0,len(itensPk)-1)])
        P0.append(Pattern(list(Vaux),0))
        P0[i+(9*tamanhoPopulacao//10)].fit = WRAcc(P0[i+(9*tamanhoPopulacao//10)].lista)
    return P0

def torneioBinario(tamanhoPopulacao,P):
    indices = []
    for i in range(tamanhoPopulacao):
        indiceP1 = randint(0,(len(P)-1))
        indiceP2 = randint(0,(len(P)-1))   
        if(abs(P[indiceP1].fit) > abs(P[indiceP2].fit)):
            indices.append(indiceP1)
        else:
            indices.append(indiceP2)
    return indices

def CRUZAMENTO_AND_2_POP(P1,P2):
    tamanhoPopulacao = len(P1)
    Pnovo = []
    indicesP1 = torneioBinario(tamanhoPopulacao, P1)
    indicesP2 = torneioBinario(tamanhoPopulacao, P2)

    for i in range(tamanhoPopulacao):
        p1 = P1[indicesP1[i]].lista
        p2 = P2[indicesP2[i]].lista
        p3 = p1 + p2
        Pnovo.append(Pattern(p3,0))
        Pnovo[i].fit = WRAcc(Pnovo[i].lista)

    return Pnovo

def CRUZAMENTO_UNIFORME_2_INDV(p1,p2):
    aux=set()
    aux2=set()
    novosPattern = []
    for i in range(len(p1.lista)):
        if bool(getrandbits(1)):
            aux.add(p1.lista[i])
        elif len(p2.lista) > i:
            aux.add(p2.lista[i])

    for i in range(len(p2.lista)):
        if bool(getrandbits(1)):
            aux2.add(p2.lista[i])
        elif len(p1.lista) > i:
            aux2.add(p1.lista[i])
    
    novosPattern.append(Pattern(list(aux),0))
    novosPattern[0].fit = WRAcc(novosPattern[0].lista)
    novosPattern.append(Pattern(list(aux2),0))
    novosPattern[1].fit = WRAcc(novosPattern[1].lista)
    return novosPattern

def MUTACAO(p):
    if not p.lista:
        aux = []
        aux.append(Pattern([randint(0,(2*len(columns)-1))],0))
        aux[0].fit = WRAcc(aux[0].lista)
        return aux[0]
    r = uniform(0,1)
    pNovo = set()
    if r < 0.33 and (len(p.lista) > 1):
        p.lista.pop(randint(0, (len(p.lista)-1)))
    elif r > 0.66:
        p.lista.pop(randint(0, (len(p.lista)-1)))
        p.lista.append(randint(0,(2*len(columns)-1)))
    else:
        p.lista.append(randint(0,(2*len(columns)-1)))
    pNovo.add(Pattern(p.lista,0))
    pNovo = list(pNovo)
    pNovo[0].fit = WRAcc(pNovo[0].lista)
    return pNovo[0]

def CRUZAMENTO_UNIFORME_2_POP(P,taxaMutacao):
    tamanhoPopulacao = len(P)
    Pnovo = []
    selecao = torneioBinario(tamanhoPopulacao, P)
    indiceSelecao = 0
    indicePnovo = 0
    while(indicePnovo < (tamanhoPopulacao-1)):
        if(uniform(0, 1) > taxaMutacao):
            novos = []
            novos = CRUZAMENTO_UNIFORME_2_INDV(P[selecao[indiceSelecao]], P[selecao[indiceSelecao+1]])
            indiceSelecao = indiceSelecao + 2
            Pnovo.append(novos[0])
            indicePnovo+=1
            if(indicePnovo < tamanhoPopulacao):
                Pnovo.append(novos[1])
                indicePnovo+=1
        else:
            Pnovo.append(MUTACAO(P[selecao[indiceSelecao]]))
            indicePnovo+= 1
            indiceSelecao+=1
    
    if(indicePnovo < tamanhoPopulacao):
        Pnovo.append(MUTACAO(P[selecao[indiceSelecao]]))
        indiceSelecao+=1
    return Pnovo

def SELECAO_MELHORES(P, Pnovo):
        tamanhoPopulacao = len(P)
        PAsterisco = []       
        PAuxiliar = []       
        PAuxiliar = (P + Pnovo)
        PAuxiliar = sorted(PAuxiliar, key=lambda x: abs(x.fit), reverse=True)
        PAsterisco = PAuxiliar[0:len(P)] 
        return PAsterisco

def SELECAO_SALVE_RELEVANTES(Pk, PAsterisco):
    indiceP = 0
    novosk = 0
    while indiceP < len(PAsterisco) and (abs(PAsterisco[indiceP].fit) > abs(Pk[len(Pk)-1].fit)):
        if E_RELEVANTE(PAsterisco[indiceP],Pk):
            Pk[len(Pk)-1] = PAsterisco[indiceP]
            Pk = sorted(Pk, key=lambda x: abs(x.fit), reverse=True)
            novosk +=1
        indiceP +=1       
    return novosk,Pk

def E_RELEVANTE(p, Pk):
    for i in range(len(Pk)):
        if(sobrescreve(p,Pk[i]) != -1):
            return False
    return True

def sobrescreve(p, Pk):
    if(sobrescreveP(p, Pk) & sobrescreveN(p, Pk)):
        if(equivalente(p, Pk)):
            return 0
        else: 
            return 1
    else:
        return -1

def sobrescreveP(p, Pk): 
    auxp = np.ones(D, dtype=np.int64)
    auxpk = np.ones(D, dtype=np.int64)
    
    for i in p.lista:
        auxp = matrz[i] & auxp
    for i in Pk.lista:
        auxpk = matrz[i] & auxpk
    for i in range(Dmais):
        if auxp[i] == 1 and auxpk[i] == 0:
            return False
    return True

def sobrescreveN(p, Pk):
    auxp  = np.ones(D, dtype=np.int64)
    auxpk = np.ones(D, dtype=np.int64)
    for i in p.lista:
        auxp = matrz[i] & auxp
    for i in Pk.lista:
        auxpk = matrz[i] & auxpk
    for i in range(Dmais,D):
        if auxp[i] == 0 and auxpk[i] == 1:
            return False
    return True
        
def equivalente(p, Pk):
    auxp = np.ones(D, dtype=np.int64)
    auxpk = np.ones(D, dtype=np.int64)
    for i in p.lista:
        auxp = matrz[i] & auxp
    for i in Pk.lista:
        auxpk = matrz[i] & auxpk
    for i in range(Dmais):#P
        if auxp[i] != auxpk[i]:
            return False
    for i in range(Dmais,D):#N
        if auxp[i] !=auxpk[i]:
            return False
    return True

def SSDP_MxC_Auto_3x3(k):
    start = time.time()
    Pk = np.empty(k)
    P = []

    Paux = INICIALIZAR_D1()

    if(len(Paux)<k):
        P = np.empty(k)
        for i in range(k):
            if(i < len(Paux)):
                P[i] = Paux[i]
            else:
                P[i] = Paux[randint(0,len(Paux)-1)]
    else:
        P = Paux

    P = sorted(P, key=lambda x: abs(x.fit), reverse=True)
    Pk = copy.deepcopy(P[0:k])

    numeroGeracoesSemMelhoraPk = 0
    indiceGeracoes = 1

    Pnovo = []
    PAsterisco = []

    tamanhoPopulacao = len(P)

    for numeroReinicializacoes in range(3):
        if(numeroReinicializacoes > 0):
            P = INICIALIZAR_ALEATORIO1_D_Pk(tamanhoPopulacao,Pk)
    
        mutationTax = 0.4
    
        while(numeroGeracoesSemMelhoraPk < 3):
            if(indiceGeracoes == 1):
                Pnovo = CRUZAMENTO_AND_2_POP(P, P)
                indiceGeracoes=indiceGeracoes+1
            else:
                Pnovo = CRUZAMENTO_UNIFORME_2_POP(P, mutationTax)

            PAsterisco = SELECAO_MELHORES(P, Pnovo)
            P = PAsterisco
         
            novosK, Pk = SELECAO_SALVE_RELEVANTES(Pk, PAsterisco)

            if(novosK > 0 and mutationTax > 0.0):
                mutationTax = mutationTax - 0.2
            elif(novosK == 0 and mutationTax < 1.0):
                mutationTax = mutationTax + 0.2
        
            if(novosK == 0 and mutationTax == 1.0):
                numeroGeracoesSemMelhoraPk = numeroGeracoesSemMelhoraPk + 1
            else:
                numeroGeracoesSemMelhoraPk = 0

        numeroGeracoesSemMelhoraPk = 0
    end = time.time()
    print("TEMPO DE EXECUCAO: ")
    print(end - start)
    for i in range(len(Pk)):
        print(Pk[i].lista)
    for i in range(len(Pk)):
        print(Pk[i].fit)

#----------------------CRIA TABELA 2 ----------------------
df = pd.read_csv('christensen-pn-freq-2.csv')
index = df.index
columns = df.columns[:-1]
values = df.values
df = df.sort_values(by=['y'], ascending=False)
df = df.reset_index(drop=True)
D = len(index)
Dmais = len(df.loc[(df['y'] == 'p')].index)
f1 = []
f2 = []

for c in columns:
    f1.append(c)
    f2.append(df[c].min())
    f1.append(c)
    f2.append(df[c].max())

Itens = pd.DataFrame()
zipped = zip(f1, f2)
Itens['attr/value'] = list(zipped)

matrz = np.full((len(Itens), len(df)), False)
df_MIN = []

for i in columns:
    df_MIN.append(df[i].min())

for i in index:
    l=0
    j=0
    for c in columns:
        if(df.loc[i, c]==df_MIN[j]):
            matrz[l][i]=1
            l=l+2
            j=j+1
        else:
            matrz[l+1][i]=1
            l=l+2
            j=j+1

SSDP_MxC_Auto_3x3(10)