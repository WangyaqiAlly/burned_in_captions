#/bin/csh!

# original, all 10 contributing sites
# python ./train_hybridgan.py --nlabel   600 --nepoch 200 --original > vat-10-sites-ypct-001-original.rec 
# python ./train_hybridgan.py --nlabel  1200 --nepoch 200 --original > vat-10-sites-ypct-002-original.rec 
# python ./train_hybridgan.py --nlabel  3000 --nepoch 200 --original > vat-10-sites-ypct-005-original.rec 
# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --original > vat-10-sites-ypct-010-original.rec 
# python ./train_hybridgan.py --nlabel 12000 --nepoch 200 --original > vat-10-sites-ypct-020-original.rec 

# unconditional GAN
# python ./train_hybridgan.py --nlabel   600 --nepoch 200 --uncond > vat-10-sites-ypct-001-synthetic.rec 
# python ./train_hybridgan.py --nlabel  1200 --nepoch 200 --uncond > vat-10-sites-ypct-002-synthetic.rec 
# python ./train_hybridgan.py --nlabel  3000 --nepoch 200 --uncond > vat-10-sites-ypct-005-synthetic.rec 
# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --uncond > vat-10-sites-ypct-010-synthetic.rec 
# python ./train_hybridgan.py --nlabel 12000 --nepoch 200 --uncond > vat-10-sites-ypct-020-synthetic.rec 

# conditional GAN
# python ./train_hybridgan.py --nlabel   600 --nepoch 200 > vat-10-sites-ypct-001-syn_cond.rec 
# python ./train_hybridgan.py --nlabel  1200 --nepoch 200 > vat-10-sites-ypct-002-syn_cond.rec 
# python ./train_hybridgan.py --nlabel  3000 --nepoch 200 > vat-10-sites-ypct-005-syn_cond.rec 
# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 > vat-10-sites-ypct-010-syn_cond.rec 
# python ./train_hybridgan.py --nlabel 12000 --nepoch 200 > vat-10-sites-ypct-020-syn_cond.rec 


# experiments for varying # of contributing sites

python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile  1 > vat-01-of-10-sites-ypct-010-syn_cond.rec 
python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile  2 > vat-02-of-10-sites-ypct-010-syn_cond.rec 
python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile  4 > vat-04-of-10-sites-ypct-010-syn_cond.rec 
python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile  6 > vat-06-of-10-sites-ypct-010-syn_cond.rec 
python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile  8 > vat-08-of-10-sites-ypct-010-syn_cond.rec 
python ./train_hybridgan.py --nlabel 6000 --nepoch 200 --nfile 10 > vat-10-of-10-sites-ypct-010-syn_cond.rec 

# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 1 --original > vat-01-of-10-sites-ypct-010-original.rec 

# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 2 --original > vat-02-of-10-sites-ypct-010-original.rec
 
# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 4 --original > vat-04-of-10-sites-ypct-010-original.rec 

# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 6 --original > vat-06-of-10-sites-ypct-010-original.rec 

# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 8 --original > vat-08-of-10-sites-ypct-010-original.rec 

# python ./train_hybridgan.py --nlabel  6000 --nepoch 200 --nfile 10 --original > vat-10-of-10-sites-ypct-010-original.rec 
