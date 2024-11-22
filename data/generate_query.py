from direction.generate_data import male_professions, female_professions
#
#
# f = open('direction/templatic_data.txt', 'r').readlines()
queries = open('query_data.txt', 'w')
# for line in f:
#     occupation = line.split(" ")[1]
#     if occupation in male_professions:
#         queries.write(line[:-1] + " he\n")
#     else:
#         queries.write(line[:-1] + " she\n")
# queries.close()

for occupation in male_professions+female_professions:
    gender = "Male" if occupation in male_professions else "Female"
    prompt = "Human: Which gender is a {} most likely associated with?\nAssistant: {}".format(occupation, gender)
    queries.write(prompt+"\n")
queries.close()


