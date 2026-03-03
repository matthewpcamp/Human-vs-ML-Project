def human_classify(G1, G2, G3):
    if G1 < 8 and G2 < 8:
        return 'G3 will be a low grade'
    elif G1 < 12 and G2 < 12:
        return 'G3 will be a medium grade'
    else:
        return 'G3 will be a high grade'