import cadquery
cadquery.Workplane('XY').box(1,2,3).toSvg()