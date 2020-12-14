# -*- coding: utf-8 -*-
english_labels = ["'",'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
 'Z']
english_lowercase_labels = [s.lower() for s in english_labels]

hebrew_labels = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל',
 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת']

farsi_labels = ['\u0623','\u0627','\u0628','\u067e','\u062a','\u062b','\u062c',
'\u0686','\u062d','\u062e','\u062f','\u0630','\u0631','\u0632','\u0698',
'\u0633','\u0634','\u0635','\u0636','\u0637','\u0638','\u0639','\u063A',
'\u0641','\u0642','\u06a9','\u06af','\u0644','\u0645','\u0646','\u0648',
'\u0647','\u06cc']


labels_map = {'english':english_labels,'hebrew':hebrew_labels,
              'farsi':farsi_labels,'english_lowercase':english_lowercase_labels}
for lang in labels_map:
    labels = labels_map[lang]
    labels.insert(0,'_') # CTC blank label. By default, blank index is 0.
    labels.append(' ')