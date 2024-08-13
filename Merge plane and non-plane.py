
files = ['cathedral1(7 48)p.txt', 'cathedral1(7 50)0.5p.txt', 'cathedral1(7 51)0.3p.txt',  'cathedral1(6 49)0.2p.txt']

with open('combined1_cathedral1 p (7 48).txt', 'wb') as f_combined:
  for f in files:
    with open(f, 'rb') as f_individual:
      f_combined.write(f_individual.read())
