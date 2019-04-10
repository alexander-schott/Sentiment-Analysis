
"""
Ran into trouble with multi word phrases in the NRC-VAD lexicon dataset.
document has format
word 

"""
with open('NRC-VAD-Lexicon.txt', 'r') as file:
    # read a list of lines into data
    data = file.readlines()


for l in range(1, len(data)):
    spaces = 0
    valPhase = 0
    line = list(data[l])
    for i in range(len(line)):
        if line[i].isalpha():
            if spaces == 1:
                line[i - 1] = '-'
        elif line[i] == ' ':
            spaces += 1
        elif line[i].isalnum():
            valPhase = 1
        if spaces >= 0 or valPhase:
            break
    data[l] = str(line)


# now change the 2nd line, note that you have to add a newline
#data[1] = 'Mage\n'

# and write everything back
with open('NRC-VAD-Lexicon.txt', 'w') as file:
    file.writelines( data )