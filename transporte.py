import numpy as np
import matplotlib.pyplot as plt

def ImprimirTabela(solucoes, nome_arquivo, iteracao, limpar=False):
    mode = 'w' if limpar else 'a'
    with open(nome_arquivo, mode) as f:
        f.write(f"Iteração {iteracao}:\n")
        np.savetxt(f, solucoes, fmt='%2d')
        f.write("\n")

def verificar_coluna(matriz, col):
    coluna = matriz[:, col]
    
    unico_um = np.sum(coluna == 1) == 1
    resto_zeros = np.sum(coluna == 0) == (coluna.size - 1)
    
    return unico_um and resto_zeros

def pivotear(solucoes, c, A, B, pv_file):
    custo = 0
    iteracao = 0
    limpar = True

    solucoes_unidimensional = solucoes.flatten()
    indices_de_variaveis_basicas = np.where(solucoes_unidimensional > 0)[0]

    tabela = np.hstack((A, B.reshape(-1, 1)))
    ImprimirTabela(tabela, pv_file, iteracao, limpar)

    linhas_pivos_disponiveis = np.arange(B.size)
    
    for ind1 in indices_de_variaveis_basicas:
        if not verificar_coluna(A, ind1):
            limpar = False
            iteracao += 1

            indices_com_um = np.where(A[:, ind1] == 1)[0]
            linhas_pivos_disponiveis_com_um = np.intersect1d(indices_com_um, linhas_pivos_disponiveis)

            B_filtrado = B[linhas_pivos_disponiveis_com_um]
            menor_indice_filtrado = np.argmin(B_filtrado)    
            ind2 = linhas_pivos_disponiveis_com_um[menor_indice_filtrado]

            indices_non_zero = np.nonzero(A[:, ind1])[0]
            indices_a_zerar = indices_non_zero.tolist() 
            indices_a_zerar.remove(ind2)

            linhas_pivos_disponiveis = linhas_pivos_disponiveis[linhas_pivos_disponiveis != ind2]

            for i in indices_a_zerar:
                fator = A[i, ind1]
                A[i, :] -= A[ind2, :] * fator
                B[i] -= B[ind2] * fator
            
            tabela = np.hstack((A, B.reshape(-1, 1)))
            ImprimirTabela(tabela, pv_file, iteracao, limpar)            

    for ind1 in indices_de_variaveis_basicas:
        if(c[ind1] != 0):
            fator = c[ind1]
            ind2 = np.argmax(A[:, ind1])
            c -= A[ind2, :]*fator
            custo -= B[ind2]*fator

    return custo

def preparar_restricoes(custos):
    n_origem, n_destino = custos.shape
    c = custos.flatten()

    A = []
    B = []

    # Restrições de oferta
    for i in range(n_origem):
        restricao = [0] * (n_origem * n_destino)
        for j in range(n_destino):
            restricao[i * n_destino + j] = 1
        A.append(restricao)
        B.append(ofertas[i])

    # Restrições de demanda
    for j in range(n_destino):
        restricao = [0] * (n_origem * n_destino)
        for i in range(n_origem):
            restricao[i * n_destino + j] = 1
        A.append(restricao)
        B.append(demandas[j])

    A = np.array(A)
    B = np.array(B)
    return c, A, B

def gsi_RN(solucoes, ofertas, demandas, nome_arquivo):
    iteracao = 0
    ofertas_copia = ofertas.copy()
    demandas_copia = demandas.copy()
    limpar = True  
    for linha in range(ofertas.size):
        for coluna in range(demandas.size):
            solucoes_anteriores = solucoes.copy()
            if ofertas_copia[linha] >= demandas_copia[coluna]:
                solucoes[linha, coluna] = demandas_copia[coluna]
                ofertas_copia[linha] -= demandas_copia[coluna]
                demandas_copia[coluna] = 0
            else:
                solucoes[linha, coluna] = ofertas_copia[linha]
                demandas_copia[coluna] -= ofertas_copia[linha]
                ofertas_copia[linha] = 0
            if not np.array_equal(solucoes, solucoes_anteriores):
                iteracao += 1
                ImprimirTabela(solucoes, nome_arquivo, iteracao, limpar)
                limpar = False  
                solucoes_anteriores = solucoes.copy()

def gsi_CM(custos, solucoes, ofertas, demandas, nome_arquivo):
    iteracao = 0
    custos_copia = custos.copy()
    demandas_copia = demandas.copy()
    ofertas_copia = ofertas.copy()
    limpar = True  
    while np.any(demandas_copia > 0) and np.any(ofertas_copia > 0):
        solucoes_anteriores = solucoes.copy()
        i_min = np.argmin(custos_copia)
        pos = np.unravel_index(i_min, custos_copia.shape)
        linha, coluna = pos
        if ofertas_copia[linha] >= demandas_copia[coluna]:
            solucoes[linha, coluna] = demandas_copia[coluna]
            ofertas_copia[linha] -= demandas_copia[coluna]
            demandas_copia[coluna] = 0
        else:
            solucoes[linha, coluna] = ofertas_copia[linha]
            demandas_copia[coluna] -= ofertas_copia[linha]
            ofertas_copia[linha] = 0
        custos_copia[linha, coluna] = 100000000
        if not np.array_equal(solucoes, solucoes_anteriores):
            iteracao += 1
            ImprimirTabela(solucoes, nome_arquivo, iteracao, limpar)
            limpar = False  
            solucoes_anteriores = solucoes.copy()

def simplex(custo, c, A, B):
    iteracao = 0
    while np.any(c > 0):
        iteracao += 1
        ind1 = np.argmax(c)
        vetor_de_indices = np.where(A[:, ind1] == 1)[0]
        valores_de_b = B[vetor_de_indices]
        indice_valor_minimo = np.argmin(valores_de_b)
        ind2 = vetor_de_indices[indice_valor_minimo]

        vet_nao_nulos = np.where(A[:, ind1] != 0)[0]
        copia = vet_nao_nulos.tolist() 
        copia.remove(ind2)

        custo -= c[ind1] * B[ind2]
        c -= c[ind1] * A[ind2, :]
        for i in copia:
            fator = A[i, ind1]
            A[i, :] -= A[ind2, :] * fator
            B[i] -= B[ind2] * fator
    
    return custo

custos = np.array([[20, 15, 10], [12, 8, 16]])
solucoes = np.zeros_like(custos)
demandas = np.array([20, 40, 60])
ofertas = np.array([50, 70])
cm_file = "CustoMinimo.txt"
rn_file = "RegraNoroeste.txt"
pv_file = "Pivoteamento.txt"
simplex_file = "Simplex.txt"

#gsi_CM(custos, solucoes, ofertas, demandas, cm_file)
gsi_RN(solucoes, ofertas, demandas, rn_file)

c_, A, B = preparar_restricoes(custos)
c = c_ * -1

custo = pivotear(solucoes, c, A, B, pv_file)

resultado = simplex(custo, c, A, B)

print(resultado)
