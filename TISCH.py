
TISCH1 = {
    'Blood':['AEL_GSE142213','ALL_GSE132509','AML_GSE116256','AML_GSE147989','CLL_GSE111014','CLL_GSE125881','PBMC_30K_10X','PBMC_60K_10X','PBMC_8K_10X'],
    'Bone':['MM_GSE117156','MM_GSE141299'],
    'Brain':['Glioma_GSE102130','Glioma_GSE103224','Glioma_GSE131928_10X','Glioma_GSE131928_Smartseq2','Glioma_GSE135437','Glioma_GSE138794','Glioma_GSE139448','Glioma_GSE141982','Glioma_GSE148842','Glioma_GSE70630','Glioma_GSE84465','Glioma_GSE89567','MB_GSE119926'],
    'Breast':['BRCA_GSE110686','BRCA_GSE114727_10X','BRCA_GSE114727_inDrop','BRCA_GSE143423','BRCA_SRP114962','BRCA_GSE138536'],
    'Colorectal':['CRC_GSE108989','CRC_GSE136394','CRC_GSE139555','CRC_GSE146771_10X','CRC_GSE146771_Smartseq2'],
    'Eye':['UVM_GSE139829'],
    'Head_Neck':['HNSC_GSE103322','HNSC_GSE139324'],
    'Liver':['CHOL_GSE125449_aPD1aPDL1aCTLA4','LIHC_GSE125449_aPDL1aCTLA4','LIHC_GSE140228_10X','LIHC_GSE140228_Smartseq2','LIHC_GSE98638'],
    'Lung':['NSCLC_EMTAB6149','NSCLC_GSE117570','NSCLC_GSE127465','NSCLC_GSE127471','NSCLC_GSE139555','NSCLC_GSE143423','NSCLC_GSE99254','NSCLC_GSE131907'],
    'Nervous':['NET_GSE140312'],
    'Pancreas':['PAAD_CRA001160', 'PAAD_GSE111672'],
    'Pelvic_cavity':['OV_GSE115007','OV_GSE118828','UCEC_GSE139555'],
    'Skin':['BCC_GSE123813_aPD1','MCC_GSE117988_aPD1aCTLA4','MCC_GSE118056_aPDL1','SCC_GSE123813_aPD1','SKCM_GSE115978_aPD1','SKCM_GSE120575_aPD1aCTLA4','SKCM_GSE123139','SKCM_GSE139249','SKCM_GSE148190','SKCM_GSE72056'],
    'STAD':['STAD_GSE134520']
    }

fliter_TISCH1 = {
    'Blood':['AEL_GSE142213','ALL_GSE132509','AML_GSE116256','CLL_GSE111014','CLL_GSE125881','PBMC_30K_10X','PBMC_60K_10X','PBMC_8K_10X'],
    'Bone':['MM_GSE117156','MM_GSE141299'],
    'Brain':['Glioma_GSE102130','Glioma_GSE103224','Glioma_GSE131928_10X','Glioma_GSE131928_Smartseq2','Glioma_GSE135437','Glioma_GSE139448','Glioma_GSE141982','Glioma_GSE148842','Glioma_GSE70630','Glioma_GSE84465','Glioma_GSE89567','MB_GSE119926'],
    'Breast':['BRCA_GSE110686','BRCA_GSE114727_10X','BRCA_GSE114727_inDrop','BRCA_GSE143423','BRCA_SRP114962'],
    'Colorectal':['CRC_GSE108989','CRC_GSE136394','CRC_GSE139555','CRC_GSE146771_10X','CRC_GSE146771_Smartseq2'],
    'Eye':['UVM_GSE139829'],
    'Head_Neck':['HNSC_GSE103322','HNSC_GSE139324'],
    'Liver':['CHOL_GSE125449_aPD1aPDL1aCTLA4','LIHC_GSE125449_aPDL1aCTLA4','LIHC_GSE140228_10X','LIHC_GSE140228_Smartseq2','LIHC_GSE98638'],
    'Lung':['NSCLC_EMTAB6149','NSCLC_GSE117570','NSCLC_GSE127465','NSCLC_GSE127471','NSCLC_GSE139555','NSCLC_GSE143423','NSCLC_GSE99254'],
    'Nervous':['NET_GSE140312'],
    'Pancreas':['PAAD_CRA001160','PAAD_GSE111672'],
    'Pelvic_cavity':['OV_GSE115007','OV_GSE118828','UCEC_GSE139555'],
    'Skin':['BCC_GSE123813_aPD1','MCC_GSE117988_aPD1aCTLA4','MCC_GSE118056_aPDL1','SCC_GSE123813_aPD1','SKCM_GSE115978_aPD1','SKCM_GSE120575_aPD1aCTLA4','SKCM_GSE123139','SKCM_GSE139249','SKCM_GSE148190','SKCM_GSE72056']
}

TISCH2_ood_test = {
    'Bone':['OS_GSE162454'],
    'Head_Neck':['LSCC_GSE150321'], 
    'Liver':['HB_GSE180665'],
    'Pelvic_cavity':['CESC_GSE168652'],
    'Skin':['MF_GSE165623']
}


TISCH1_cancer_type_map = {'normal': 0.0, 'AEL': 1.0, 'ALL': 2.0, 'AML': 3.0, 'CLL': 4.0, 'PBMC': 5.0, 'MM': 6.0, 'Glioma': 7.0, 'MB': 8.0, 'BRCA': 9.0, 'CRC': 10.0, 'UVM': 11.0, 'HNSC': 12.0, 'CHOL': 13.0, 'LIHC': 14.0, 'NSCLC': 15.0, 'NET': 16.0, 'PAAD': 17.0, 'OV': 18.0, 'UCEC': 19.0, 'BCC': 20.0, 'MCC': 21.0, 'SCC': 22.0, 'SKCM': 23.0, 'STAD': 24.0}

TISCH1_tissue_map = {'Blood': 0.0, 'Bone': 1.0, 'Brain': 2.0, 'Breast': 3.0, 'Colorectal': 4.0, 'Eye': 5.0, 'Head_Neck': 6.0, 'Liver': 7.0, 'Lung': 8.0, 'Nervous': 9.0, 'Pancreas': 10.0, 'Pelvic_cavity': 11.0, 'Skin': 12.0, 'STAD':13.0}