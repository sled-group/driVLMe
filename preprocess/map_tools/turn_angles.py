

town01_intersection_triples = {}

town02_intersection_triples = {}

town03_intersection_triples = {
    # intersection 1932
    (238, 1932, 82) : 'straight', 
    (82, 1932, 238) : 'straight',
    (82, 1932, 1019) : 'right',
    (1019, 1932, 82) : 'left',
    (82, 1019, 1932) : 'left',

    # intersection 1901
    (1669, 1901, 861) : 'right', 
    (1820, 1901, 861) : 'left',
    (861, 1901, 1820) : 'right',
    (861, 1901, 1669) : 'left',

    # intersection 861
    (1820, 861, 1901) : 'sharp right',
    (1901, 861, 1820) : 'sharp left',
    (1820, 861, 238) : 'gentle right',
    (238, 861, 1820) : 'gentle left',
    (1901, 861, 730) : 'slight left',
    (730, 861, 1901) : 'gentle right',
    (1901, 861, 238) : 'sharp right',
    (238, 861, 1901) : 'sharp left',
    (1901, 861, 1682) : 'gentle right',
    (1682, 861, 1901) : 'gentle left'
}

town05_intersection_triples = {}

intersection_triples = {
    'Town01' : town01_intersection_triples,
    'Town02' : town02_intersection_triples,
    'Town03' : town03_intersection_triples,
    'Town05' : town05_intersection_triples,
}