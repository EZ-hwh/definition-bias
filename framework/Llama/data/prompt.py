NER_prompt = '''Please list all entity words in the text that fit the category. Here\'s the category list: 
{0}
And then output the result in the format of ```type1: entity1; type2: entity2; ...```'''

RE_prompt = '''Given a sentence or paragraph, and a given relationship set that describe the relation between them. Here\'s the relation set:
{0}
Output the result in the format of ```(subject1, relation1, object1), (subject2, relation2, object2), ...```'''