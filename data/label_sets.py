# -*- coding: utf-8 -*-
english_labels = ["'",'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
 'Z']
english_lowercase_labels = [s.lower() for s in english_labels]

hebrew_labels = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל',
 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת']

labels_map = {'english':english_labels,'hebrew':hebrew_labels,'english_lowercase':english_lowercase_labels}
for lang in labels_map:
    labels = labels_map[lang]
    labels.insert(0,'_') # CTC blank label. By default, blank index is 0.
    labels.append(' ')