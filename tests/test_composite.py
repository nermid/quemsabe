import numpy as np
from PIL import Image
from composite import render, extract

P = dict(x=16,y=16,scale=1.,rotation=0.,brightness=1.,contrast=1.,saturation=1.,rgb_gain=[1.,1.,1.])

def test_original_opaque_pixels_and_background():
    bg = Image.new('RGB',(32,32),(20,40,60))
    rgb = np.full((32,32,3),[110,150,180],np.uint8)
    a = np.zeros((32,32)); a[8:24,8:24] = 1
    result = np.asarray(render(bg,rgb,a,P))
    assert np.max(abs(result[12,12].astype(int)-rgb[12,12])) <= 1
    assert np.max(abs(result[0,0].astype(int)-np.array([20,40,60]))) <= 1

def test_manual_matting(tmp_path):
    rgb = np.zeros((32,32,3),np.uint8); rgb[:]=[30,40,50]; rgb[8:24,8:24]=[160,100,80]
    tri = np.zeros((32,32),np.uint8); tri[7:25,7:25]=128; tri[9:23,9:23]=255
    path=tmp_path/'trimap.png'; Image.fromarray(tri).save(path)
    fg,a,_ = extract(Image.fromarray(rgb),None,path)
    assert a[16,16] > .99 and a[0,0] < .01
    assert np.array_equal(fg[16,16],rgb[16,16])

def test_cli_exports(tmp_path):
    import subprocess
    import sys
    import json
    bg=tmp_path/'background.jpg'; src=tmp_path/'subject.jpg'; tri=tmp_path/'trimap.png'
    Image.new('RGB',(64,80),(70,90,110)).save(bg)
    a=np.full((32,32,3),[20,40,60],np.uint8); a[8:24,8:24]=[170,110,90]
    Image.fromarray(a).save(src)
    t=np.zeros((32,32),np.uint8); t[7:25,7:25]=128; t[9:23,9:23]=255
    Image.fromarray(t).save(tri)
    out=tmp_path/'output'
    subprocess.run([sys.executable,'composite.py','--subject',str(src),'--background',str(bg),'--trimap',str(tri),'--out',str(out)],check=True)
    assert Image.open(out/'portrait.jpg').size == (64,80)
    assert Image.open(out/'subject.png').mode == 'RGBA'
    assert Image.open(out/'mask.png').size == (32,32)
    assert json.loads((out/'parameters.json').read_text())['scale'] == 1
