setting=$1
run=$2

for attn in so sp ts ; do
    for out in so sp ts ; do
        python translate.py -model $( python select_model.py < /mnt/data/bpop/acl2019/morph/$setting/multi/$attn-$out-logs/$run.log ) -src /mnt/data/bpop/acl2019/morph/$setting/multi/data/test.src -beam_size 5 -gpu 0 -output /mnt/data/bpop/acl2019/morph/$setting/multi/$attn-$out-test-out/$run.pred
    done
done
