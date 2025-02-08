NER_prompt = '''Here\'s a named entity recognition task from dataset {0}.
Please list all entity words in the text that fit the category \'{1}\'.
Output the result in the format of ```type1: entity1; type2: entity2; ...```'''

RE_prompt = '''Here\'s a relation extraction task from dataset {0}.
Given a sentence or paragraph, please recognize the entity in it and extract triples that describe the \'{1}\' relation between them. 
Output the result in the format of ```(subject1, relation1, object1), (subject2, relation2, object2), ...```'''