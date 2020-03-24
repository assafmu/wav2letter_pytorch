# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:02:23 2020

@author: User
"""
hebrew_replacements = [('צ ','ץ '),
                       ('פ ','ף '),
                       ('כ ','ך '),
                       ('מ ','ם '),
                       ('נ ','ן ')]
def hebrew_normal_to_final(strings):
    return _hebrew_convert(strings,hebrew_replacements)

def hebrew_final_to_normal(strings):
    return _hebrew_convert(strings,{(k[1],k[0])for k in hebrew_replacements})

def _hebrew_convert(strings,replacements):
    if type(strings) is list:
        return [hebrew_normal_to_final(s) for s in strings]

    res = strings + ' '
    for replacement in replacements:
        res = res.replace(replacement[0],replacement[1])
    return res[:-1]

