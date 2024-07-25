import numpy as np

def ImprimirTabela(solucoes, nome_arquivo, iteracao, limpar=False):
    mode = 'w' if limpar else 'a'
    with open(nome_arquivo, mode) as f:
        f.write(f"Iteração {iteracao}:\n")
        np.savetxt(f, solucoes, fmt='%2d')
        f.write("\n")


def verificar_coluna(A, col):
    coluna = A[:, col]
    
    unico_um = np.sum(coluna == 1) == 1
    resto_zeros = np.sum(coluna == 0) == (coluna.size - 1)
    
    return unico_um and resto_zeros

def pivotear(solucoes, c, A, B):
    custo = 0
    iteracao = 0
    limpar = True

    solucoes_unidimensional = solucoes.flatten()
    indices_de_variaveis_basicas = np.where(solucoes_unidimensional > 0)[0]

    tabela = np.hstack((A, B.reshape(-1, 1)))
    ImprimirTabela(tabela, "Pivoteamento.txt", iteracao, limpar)

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

            indices_nao_nulos = np.nonzero(A[:, ind1])[0]
            indices_a_zerar = indices_nao_nulos.tolist() 
            indices_a_zerar.remove(ind2)

            linhas_pivos_disponiveis = linhas_pivos_disponiveis[linhas_pivos_disponiveis != ind2]

            for i in indices_a_zerar:
                fator = A[i, ind1]
                A[i, :] -= A[ind2, :] * fator
                B[i] -= B[ind2] * fator
            
            tabela = np.hstack((A, B.reshape(-1, 1)))
            ImprimirTabela(tabela, "Pivoteamento.txt", iteracao, limpar)            

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

def gsi_CM(custos, solucoes, ofertas, demandas):
    custos_copia = custos.copy()
    demandas_copia = demandas.copy()
    ofertas_copia = ofertas.copy()
    while np.any(demandas_copia > 0) and np.any(ofertas_copia[:-1] > 0):
        i_min = np.argmin(custos_copia[:-1], axis=None)
        pos = np.unravel_index(i_min, custos_copia[:-1].shape)
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

    for coluna in range(len(demandas_copia)):
        if demandas_copia[coluna] > 0:
            solucoes[-1, coluna] = demandas_copia[coluna]  
            demandas_copia[coluna] = 0

def simplex(custo, c, A, B):
    while np.any(c > 0):
        ind1 = np.argmax(c)
        vetor_de_indices = np.where(A[:, ind1] == 1)[0]
        valores_de_b = B[vetor_de_indices]
        indice_valor_minimo = np.argmin(valores_de_b)
        ind2 = vetor_de_indices[indice_valor_minimo]

        vet_nao_nulos = np.where(A[:, ind1] != 0)[0]
        copia = vet_nao_nulos.tolist()  # Convertendo para lista para usar remove()
        copia.remove(ind2)

        custo -= c[ind1] * B[ind2]
        c -= c[ind1] * A[ind2, :]
        for i in copia:
            fator = A[i, ind1]
            A[i, :] -= A[ind2, :] * fator
            B[i] -= B[ind2] * fator
            
    return custo

custos = np.array([[100000000, 10, 12, 8, 9, 5], 
                   [10, 100000000, 15, 16, 8, 10],
                   [12, 15, 100000000, 10, 8, 12],
                   [8, 16, 10, 100000000, 15, 5],
                   [9, 8, 8, 15, 100000000, 20],
                   [5, 10, 12, 5, 20, 100000000],
                   [0, 0, 0, 0, 0, 0]])

solucoes = np.zeros_like(custos)
demandas = np.array([600, 250, 250, 500, 150, 100])
ofertas = np.array([400, 300, 150, 300, 100, 250, 350])
nome_arquivo = "solucoes.txt"

gsi_CM(custos, solucoes, ofertas, demandas)

#print(solucoes)

c_, A, B = preparar_restricoes(custos)
c = c_ * -1

custo = pivotear(solucoes, c, A, B)

print(custo)

resultado = simplex(custo, c, A, B)

print(resultado)