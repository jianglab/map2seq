def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess
            subprocess.call(f'pip install {package_pip_name}', shell=True)
            scope[package_import_name] =  __import__(package_import_name)

import streamlit as st
import numpy as np
import re
import os
from shutil import which
from findmysequence_lib.findmysequence import fms_main
import pickle
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Label

tmpdir = "tempDir"
tmpdir = os.path.abspath(tmpdir)

def main():
    title = "Identification of Proteins from Density Map"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    #https://discuss.streamlit.io/t/hide-titles-link/19783/4
    st.markdown(""" <style> .css-15zrgzn {display: none} </style> """, unsafe_allow_html=True)

    with st.expander(label="README", expanded=False):
            st.write("Sample readme")

    col1, col2 = st.columns([1.25, 3])

    mrc = None
    pdb = None

    with col1:
        st.subheader("Settings")

        mrc = None
        # make radio display horizontal
        st.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        # input_modes_map = {0:"upload", 1:"emd-xxxxx"}
        input_modes_map = {0:"upload", 1:"url", 2:"emd-xxxxx"}
        help_map = "Only maps in MRC (*\*.mrc*) or CCP4 (*\*.map*) format are supported. Compressed maps (*\*.gz*) will be automatically decompressed"
        input_mode_map = st.radio(label="How to obtain the input map:", options=list(input_modes_map.keys()), format_func=lambda i:input_modes_map[i], index=0, help=help_map, key="input_mode_map")
        is_emd = False
        emdb_ids_all, emdb_ids_helical, methods = get_emdb_ids()
        if input_mode_map == 0: # "upload a MRC file":
                label = "Upload a map in MRC or CCP4 format"
                help = None
                fileobj = st.file_uploader(label, type=['mrc', 'map', 'map.gz'], help=help, key="file_upload")
                if fileobj is not None:
                    emd_id = extract_emd_id(fileobj.name)
                    is_emd = emd_id is not None
                    with open(os.path.join(tmpdir, fileobj.name), "wb") as f:
                        f.write(fileobj.getbuffer())
                    mrc = tmpdir + "/" + fileobj.name
        elif input_mode_map == 1: # "url":
            url_default = "https://ftp.wwpdb.org/pub/emdb/structures/EMD-10499/map/emd_10499.map.gz"
            help = "An online url (http:// or ftp://) or a local file path (/path/to/your/structure.mrc)"
            url = st.text_input(label="Input the url of a 3D map:", value=url_default, help=help, key="url")
            emd_id = extract_emd_id(url)
            is_emd = emd_id is not None and emd_id
            with st.spinner(f'Downloading {url.strip()}'):
                filepath = get_3d_map_from_url(url.strip())
                mrc = filepath
        elif input_mode_map == 2: # "emdb": randomly selects form emdb_ids_all not emdb_ids_helical anymore
            if not emdb_ids_all:
                st.warning("failed to obtained a list of helical structures in EMDB")
                return
            url = "https://www.ebi.ac.uk/emdb/search/*%20AND%20structure_determination_method:%22helical%22?rows=10&sort=release_date%20desc"
            st.markdown(f'[All {len(emdb_ids_helical)} helical structures in EMDB]({url})')
            emd_id_default = "emd-10499"
            do_random_embid = st.checkbox("Choose a random EMDB ID", value=False, key="random_embid")
            if do_random_embid:
                help = "Randomly select another helical structure in EMDB"
                # if max_map_size>0: help += f". {warning_map_size}"
                button_clicked = st.button(label="Change EMDB ID", help=help)
                if button_clicked:
                    import random
                    st.session_state.emd_id = 'emd-' + random.choice(emdb_ids_all)
            else:
                help = None
                # if max_map_size>0: help = warning_map_size
                label = "Input an EMDB ID (emd-xxxxx):"
                emd_id = st.text_input(label=label, value=emd_id_default, key='emd_id', help=help)
                emd_id = emd_id.lower().split("emd-")[-1]
                if emd_id not in emdb_ids_all:
                    import random
                    msg = f"EMD-{emd_id} is not a valid EMDB entry. Please input a valid id (for example, a randomly selected valid id 'emd-{random.choice(emdb_ids_helical)}')"
                    st.warning(msg)
                    return
                elif emd_id not in emdb_ids_helical:
                    msg= f"EMD-{emd_id} is in EMDB but annotated as a '{methods[emdb_ids_all.index(emd_id)]}' structure, not a helical structure" 
                    st.warning(msg)
            if 'emd_id' in st.session_state: emd_id = st.session_state.emd_id
            else: emd_id = emd_id_default
            emd_id = emd_id.lower().split("emd-")[-1]
            with st.spinner(f'Downloading EMD-{emd_id} from {get_emdb_map_url(emd_id)}'):
                filepath = get_emdb_map(emd_id)
                mrc = filepath
            if mrc is None:
                st.warning(f"Failed to download [EMD-{emd_id}](https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id})")
                return
            # is_emd = emd_id in emdb_ids_helical
        #----------------------------------------------------------------------------------------------------------------------------------------

        #pdb input
        input_modes_model = {0:"upload", 1:"url", 2:"PDB ID"}
        #input_modes_model = {0:"upload"}
        help_model = None
        input_mode_model = st.radio(label="How to obtain the input PDB file:", options=list(input_modes_model.keys()), format_func=lambda i:input_modes_model[i], index=0, help=help_model, key="input_mode_model")
        # pdb_ids_all = get_pdb_ids()
        pdb = None

        if input_mode_model == 0: # "upload a PDB file":
            label = "Upload a PDB file"
            fileobj = st.file_uploader(label, type=['pdb'], help=None, key="file_upload")
            if fileobj is not None:
                remove_old_pdbs()
                with open(os.path.join(tmpdir, fileobj.name), "wb") as f:
                    f.write(fileobj.getbuffer())
                pdb = tmpdir + "/" + fileobj.name
                # pdb = get_model_from_uploaded_file(fileobj)
        
        elif input_mode_model == 1: # "url":
            help = "An online url (http:// or ftp://) or a local file path (/path/to/your/model.pdb)"
            url = st.text_input(label="Input the url of a PDB model:", help=help, key="url")
            if url:
                with st.spinner(f'Downloading {url.strip()}'):
                    filepath = get_3d_map_from_url(url.strip())
                    pdb = filepath
                st.write("Done.")
        
        elif input_mode_model == 2: # "PDB ID":
            help = None
            # if max_map_size>0: help = warning_map_size
            label = "Input an PDB ID (for example: 4hhb):"
            pdb_id = st.text_input(label=label, key='pdb_id', help=help)
            pdb_id = pdb_id.lower()
            if pdb_id:
                pdb_url=get_pdb_url(pdb_id)
                with st.spinner(f'Downloading {pdb_id}.pdb from {pdb_url}'):
                    filepath = get_3d_map_from_url(pdb_url)
                    pdb = filepath
                st.write("Done.")
        
        print(pdb)
        print(mrc)
        print(which('hmmsearch'))
        print('-')
    if mrc is None: return
    if pdb is None: return

    with col2:
        st.subheader("Result Plot")
        graph_success = False
        with st.spinner("Processing..."):
            remove_old_graph_log()
            cmd_template = f"python .\main.py -map {mrc} -pdb {pdb}"
            os.system(cmd_template)
            print('Main done.')

            #with open('tempDir/log.txt') as f:
            with open(os.path.join(tmpdir, 'log.txt')) as f:
                first_line = f.readline()
                if "Success" in first_line:
                    graph_success = True
        if graph_success:
            #from PIL import Image
            #image = Image.open(os.path.join(tmpdir, 'fms.png'))
            #st.image(image, caption='Temp Title')
            with open(os.path.join(tmpdir, 'fms.png_x.pkl'),'rb') as inf:
                xs = pickle.load(inf)
            with open(os.path.join(tmpdir, 'fms.png_y.pkl'),'rb') as inf:
                ys = pickle.load(inf)
            
            #https://docs.bokeh.org/en/latest/docs/user_guide/tools.html
            
            source = ColumnDataSource(data=dict(x=range(len(xs)),y=ys,ID=xs))
            top_source = ColumnDataSource(data=dict(x=[0],y=[ys[0]],ID=[xs[0]]))
            label = Label(x=0, y=ys[0], text='Best Match', x_offset=10, y_offset=-5, render_mode='canvas')
  
            TOOLTIPS = [('index','$index'),('ID','@ID'),('E-val','@y')]
   
            p = figure(width=400,height=400,tooltips=TOOLTIPS,y_axis_type='log', title='Ranked Sequences')
            p.circle('x','y',source=source)
            p.circle('x','y',source=top_source, size=10,line_color='red',fill_color='red')
            p.yaxis.axis_label = 'E-values'
            p.xaxis.axis_label = 'Rank Order'
            p.y_range.flipped = True
            p.add_layout(label)
            
            st.bokeh_chart(p, use_container_width=True)
        else:
            st.text('Failed')

    
def remove_old_graph_log():
    dir = os.listdir(tmpdir)
    for item in dir:
        if item.endswith(".png") or item.endswith(".txt"):
            os.remove(os.path.join(tmpdir, item))

#-------------------------------Map Functions-------------------------------
def remove_old_maps():

    dir = os.listdir(tmpdir)
    for item in dir:
        if item.endswith(".mrc") or item.endswith(".map") or item.endswith(".map.gz"):
            os.remove(os.path.join(tmpdir, item))

def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

#"https://ftp.wwpdb.org/" -> threw a IsADirectoryError: This app has encountered an error. ...
#Since I delete maps from the "tempDir" folder 
@st.experimental_singleton(show_spinner=False, suppress_st_warning=True)
def get_3d_map_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    ds = np.DataSource(None)
    if not ds.exists(url_final):
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    filename_final = None
    localfile_path  = None
    with ds.open(url) as fp:
        # data = get_3d_map_from_file(fp.name)
        if fp.name.endswith(".gz"):
            filename_final = fp.name[:-3]
            import gzip, shutil
            with gzip.open(fp.name, 'r') as f_in, open(filename_final, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            filename_final = fp.name
  
        #copies mrc file from tmp directory to "tempDir" files for findmysequence are stored
        import shutil
        shutil.copy2(filename_final, tmpdir)
        #localfile_path = tmpdir + "/" + filename_final.rsplit('/', 1)[-1]
        map_name_splitted=re.split(r'/|\\',filename_final)
        localfile_path = tmpdir + "/" + map_name_splitted[-1]
    return localfile_path

def extract_emd_id(text):
    import re
    pattern = '.*emd_([0-9]*)\.map.*'

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        emd_id = match.group(1)
    else:
        emd_id = None
    return emd_id

@st.experimental_singleton(show_spinner=False)
def get_emdb_ids():
    try:
        import_with_auto_install(["pandas"])
        import pandas as pd
        entries_all = pd.read_csv('https://www.ebi.ac.uk/emdb/api/search/current_status:"REL"?wt=csv&download=true&fl=emdb_id,structure_determination_method,resolution')
        methods = list(entries_all["structure_determination_method"])
        entries_helical = entries_all[entries_all["structure_determination_method"]=="helical"]
        emdb_ids_all     = list(entries_all.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
        emdb_ids_helical = list(entries_helical.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    except:
        emdb_ids_all = []
        emdb_ids_helical = []
        methods = {}
    return emdb_ids_all, emdb_ids_helical, methods

def get_emdb_map_url(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    server = "https://ftp.wwpdb.org/pub"    # Rutgers University, USA
    #server = "https://ftp.ebi.ac.uk/pub/databases" # European Bioinformatics Institute, England
    #server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
    url = f"{server}/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return url

def get_pdb_url(protid):
	server = "https://files.rcsb.org/download"
	return f"{server}/{protid}.pdb.gz"
	
@st.experimental_singleton(show_spinner=False, suppress_st_warning=True)
def get_emdb_map(emdid):
    url = get_emdb_map_url(emdid)
    data = get_3d_map_from_url(url)
    # If no this line the returning path will be wrong. WHY???
    return data

#-------------------------------End Map Functions-------------------------------

#-------------------------------Model Functions-------------------------------
def remove_old_pdbs():
    dir = os.listdir(tmpdir)
    for item in dir:
        if item.endswith(".pdb"):
            os.remove(os.path.join(tmpdir, item))

@st.cache(persist=True, show_spinner=False, ttl=24*60*60.) # refresh every day
def get_pdb_ids():
    try:
        url = "ftp://ftp.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
        ds = np.DataSource(None)
        with ds.open(url) as fp:
            pdb_ids = [line[:4] for line in fp.readlines()[2:] if len(line) > 4]
    except:
        pdb_ids = None
    return pdb_ids
#-------------------------------End Model Functions-------------------------------

if __name__ == "__main__":  
    main()
