def human_classify(G1, G2, G3):
    if G1 < 8 and G2 < 8:
        return 'Low'
    elif G1 < 12 and G2 < 12:
        return 'Medium'
    else:
        return 'High'