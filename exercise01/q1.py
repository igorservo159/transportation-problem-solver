import numpy as np
import pandas as pd

inf = 1000000

def ImprimirTabelaEmTXT(tabela, nome_arquivo, iteracao, limpar=False):
    mode = 'w' if limpar else 'a'
    with open(nome_arquivo, mode) as f:
        f.write(f"Iteração {iteracao}:\n")
        np.savetxt(f, tabela, fmt='%2d')
        f.write("\n")

def ImprimirTabela(tabela, nome_arquivo, iteracao, limpar=False):
    df = pd.DataFrame(tabela)
    with pd.ExcelWriter(nome_arquivo, mode='a' if not limpar else 'w', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f"Iteracao_{iteracao}", index=False, header=False)

def verificar_coluna(A, col):
    coluna = A[:, col]
    
    unico_um = np.sum(coluna == 1) == 1
    resto_zeros = np.sum(coluna == 0) == (coluna.size - 1)
    
    return unico_um and resto_zeros

def pivotear(solucoes, c, A, B):
    custo = 0
    iteracao = 0
    limpar = True

    pv_file = "Pivoteamento.xlsx"
    pv_file_txt = "Pivoteamento.txt"

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

            indices_nao_nulos = np.nonzero(A[:, ind1])[0]
            indices_a_zerar = indices_nao_nulos.tolist() 
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

def gerar_solucao_inicial_por_Custo_Minimo(custos, solucoes, ofertas, demandas):
    cmc_file = "CustoMinimoCustos.xlsx"
    cms_file = "CustoMinimoSolucoes.xlsx"
    cmc_file_txt = "CustoMinimoCustos.txt"
    cms_file_txt = "CustoMinimoSolucoes.txt"

    custos_copia = custos.copy()
    demandas_copia = demandas.copy()
    ofertas_copia = ofertas.copy()

    iteracao_solucoes = 0
    iteracao_custos = 0
    limpar = True

    custos_com_demandas = np.vstack([custos_copia, demandas_copia])
    tabela_custos = np.hstack([custos_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])

    solucoes_com_demandas = np.vstack([solucoes, demandas_copia])
    tabela_solucoes = np.hstack([solucoes_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])

    ImprimirTabela(tabela_custos, cmc_file, iteracao_custos, limpar)
    ImprimirTabela(tabela_solucoes, cms_file, iteracao_solucoes, limpar)
    limpar = False
    
    while np.any(demandas_copia > 0) and np.any(ofertas_copia[:-1] > 0):
        copia_tabela_custos = tabela_custos.copy()
        copia_tabela_solucoes = tabela_solucoes.copy()

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
        custos_copia[linha, coluna] = inf

        custos_com_demandas = np.vstack([custos_copia, demandas_copia])
        tabela_custos = np.hstack([custos_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])

        solucoes_com_demandas = np.vstack([solucoes, demandas_copia])
        tabela_solucoes = np.hstack([solucoes_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])

        if not np.array_equal(copia_tabela_solucoes, tabela_solucoes):
            iteracao_solucoes += 1
            ImprimirTabela(tabela_solucoes, cms_file, iteracao_solucoes, limpar)

        if not np.array_equal(copia_tabela_custos, tabela_custos):
            iteracao_custos += 1
            ImprimirTabela(tabela_custos, cmc_file, iteracao_custos, limpar)

    for coluna in range(len(demandas_copia)):
        if demandas_copia[coluna] > 0:
            copia_tabela_custos = tabela_custos.copy()
            copia_tabela_solucoes = tabela_solucoes.copy()

            solucoes[-1, coluna] = demandas_copia[coluna]  
            ofertas_copia[-1] -= demandas_copia[coluna]
            demandas_copia[coluna] = 0

            custos_com_demandas = np.vstack([custos_copia, demandas_copia])
            tabela_custos = np.hstack([custos_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])

            solucoes_com_demandas = np.vstack([solucoes, demandas_copia])
            tabela_solucoes = np.hstack([solucoes_com_demandas, np.append(ofertas_copia, [0]).reshape(-1, 1)])
            
            if not np.array_equal(copia_tabela_solucoes, tabela_solucoes):
                iteracao_solucoes += 1
                ImprimirTabela(tabela_solucoes, cms_file, iteracao_solucoes, limpar)

            if not np.array_equal(copia_tabela_custos, tabela_custos):
                iteracao_custos += 1
                ImprimirTabela(tabela_custos, cmc_file, iteracao_custos, limpar)


def simplex(custo, c, A, B):
    iteracao = 0
    simplex_file = "Simplex.xlsx"
    fatores_file = "Fatores.txt"

    # Adiciona uma coluna a mais para c e custo
    c_expanded = np.append(c, custo)
    tabela = np.hstack((A, B.reshape(-1, 1)))
    tabela = np.vstack((c_expanded, tabela))
    limpar = True
    ImprimirTabela(tabela, simplex_file, iteracao, limpar)
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

        fator = c[ind1]
        custo -= fator * B[ind2]
        c -= fator * A[ind2, :]

        coluna_linha_pivo_fator = [ind1, 0, ind2+1, fator]
        ImprimirTabelaEmTXT(coluna_linha_pivo_fator, fatores_file, iteracao, False)
        for i in copia:
            fator = A[i, ind1]
            A[i, :] -= A[ind2, :] * fator
            B[i] -= B[ind2] * fator
            coluna_linha_pivo_fator = [ind1, i+1, ind2+1, fator]
            ImprimirTabelaEmTXT(coluna_linha_pivo_fator, fatores_file, iteracao, False)
            limpar = False

        c_expanded = np.append(c, custo)
        tabela = np.hstack((A, B.reshape(-1, 1)))
        tabela = np.vstack((c_expanded, tabela))
        ImprimirTabela(tabela, simplex_file, iteracao, False)

    return custo


custos = np.array([[inf, 10, 12, 8, 9, 5], 
                   [10, inf, 15, 16, 8, 10],
                   [12, 15, inf, 10, 8, 12],
                   [8, 16, 10, inf, 15, 5],
                   [9, 8, 8, 15, inf, 20],
                   [5, 10, 12, 5, 20, inf],
                   [0, 0, 0, 0, 0, 0]])

solucoes = np.zeros_like(custos)
demandas = np.array([600, 250, 250, 500, 150, 100])
ofertas = np.array([400, 300, 150, 300, 100, 250, 350]) 

gerar_solucao_inicial_por_Custo_Minimo(custos, solucoes, ofertas, demandas)

c_, A, B = preparar_restricoes(custos)
c = c_ * -1

custo = pivotear(solucoes, c, A, B)

A_sem_linha_degenerada = np.delete(A, 9, axis=0)
B_sem_linha_degenerada = np.delete(B, 9)

resultado = simplex(custo, c, A_sem_linha_degenerada, B_sem_linha_degenerada)

print(resultado)
