
## remove os arquivos compilados pra nao ter dados duplicados
rm datasets/oficial/compilado.csv
rm datasets/oficial/features.csv

# Treat mov Bom
# Remove a pasta de arquivos tratados, e trata eles novamente
rm -r datasets/oficial/mov_bom/david/treated
python3.5 treat_data.py datasets/oficial/mov_bom/david/ 1

rm -r datasets/oficial/mov_bom/lucas/treated
python3.5 treat_data.py datasets/oficial/mov_bom/lucas/ 1

rm -r datasets/oficial/mov_bom/martim/treated
python3.5 treat_data.py datasets/oficial/mov_bom/marim/ 1

rm -r datasets/oficial/mov_bom/raddatz/treated
python3.5 treat_data.py datasets/oficial/mov_bom/raddatz/ 1

rm -r datasets/oficial/mov_bom/ricardo/treated
python3.5 treat_data.py datasets/oficial/mov_bom/ricardo/ 1

rm -r datasets/oficial/mov_bom/schwingel/treated
python3.5 treat_data.py datasets/oficial/mov_bom/schwingel/ 1




# Treat mov ruim
rm -r datasets/oficial/mov_ruim/david/treated
python3.5 treat_data.py datasets/oficial/mov_ruim/david/ 0

rm -r datasets/oficial/mov_ruim/marim/treated
python3.5 treat_data.py datasets/oficial/mov_ruim/martim/ 0

rm -r datasets/oficial/mov_ruim/moises/treated
python3.5 treat_data.py datasets/oficial/mov_ruim/moises/ 0

rm -r datasets/oficial/mov_ruim/ricardo/treated
python3.5 treat_data.py datasets/oficial/mov_ruim/ricardo/ 0
