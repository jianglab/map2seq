# Copyright (c) 2021, EMBL/Grzegorz Chojnowski (gchojnowski@embl-hamburg.de)
# All rights reserved.
# License: BSD 3-Clause License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = "Grzegorz Chojnowski"
__date__ = "06 Jun 2020"





import sys, os, re

from shutil import which

ROOT = os.path.dirname(os.path.realpath(__file__))

import random
import subprocess
import string
import gzip
import pyhmmer

# cctbx imports
from iotbx.pdb import amino_acid_codes as aac



from scipy.special import erf,erfc
from scipy import stats

import numpy as np

from iotbx.bioinformatics import any_sequence_format

from fmslib import nn_predict
from fmslib import xyz_utils

import tempfile

random_sequence='MLFVGDDWAEDHHDLYLMNEAGDRLASRRLPEGLAGIRLLHDLIAAHADDPAQVAVGIETDRGLWVEALTGAGYQVYAVNPLAVARYRDRHAVSGAKSDAADAKLLADLVRTDRHNHRLIAGDTPDAEAVKVLARAHQNLIWTRNRHTNALRSALREYYPAALEAFDDLSDRDALAILGHAPDPRQASSLSLAKIRSALKAAGRQRNIDIRAQEIQAALRSEQLAAPAAVTAAFAATTRATVGLVVELSRQIDDLENELAAHFETHPDADIYRSLPGLGVILGARVLGEFGDDPNRYTNAKCRKNYAGTSPLTIASGRKRAVLARHIRNRRLYDAIDQWAFCALTRSPGARQFYDHRRAEGDLHHQALRALGNRLVGILHGCLRHRTRYDEHKAWAHRTPAAAMEAFEGNRAETATMLPVINAFKAAHQLTDVTVVADAGMISEANQVALQSAGLSYILGAKIPFLPDVVREWRDKHPDEAIPDGLVLTQPWPATSSEKARGIPDRVIHYQFRHDRARRTLRGIDEQVAKAQRAVDGHAPVKRNRYIQLTGATKSVNRTLEAKTRALAGWKGYTTNLVGQPASFVIDAYHQLWHIEKAFRMSKHDLQARPIYHHLRESIEAHLSIVVAAMAVSHFIETQTGWSIKKFVRTARRYRTVKIKAGTQTLTAADPLPEDLRAVLIKIRADGAHMMTVKHAVVIGAFAVAAVAGQLAVAAPAEAKRCPAGTVEKFEGVCIKGSGGGSVAPPVMAPSAGGAKIQNLPGQLPSVNGVPCTIEHYGTCLAMTQPMNEFIKIVLGSAPEQTGVATLLLSRPPTNALTRQMYREISIAANELAQRADVSSVIVYGGHEIFCAGDDIPELRTLDAEETAAADHALQRCIEAVAAIPKPTVAAVTGYALGSGMNLALAADWRVSGDNAKFGATEILAGLAPRGGGGVRLADAIGTSKAKELVFSGRFVGAEEALEIGLVDEMVAPDHVYEAALAWARRFSDHPVDVLAAAKASLNGPVNWRGPRPTDRMVDRPMFTAPAIRTSSASSSSSFASCFAAPLPRGLRAVAQEQTWDAFLARFAAIAGPVTLRQWSCVDSRPTGQIGPREYQATLSIGDTVATSSVTAYGPVAALTEILHAHGITVETTSFHQLPTRGQTATFIEGSNGVHREWAMGLDTDPVQSALRAVIACANRLTAHMAQGSDFAGKRCFVTGAASGIGRATALALAAAGAELYLTDRDADGLVQTVADARALGAEVPAHRALDISDYDQVREFAKDIHAAHGSMDVVMNIAGVSAWGTVDRLSHQQWRSMVDINLMGPIHVIEEFIPPMIEARRGGHLVNVSSAAGLVALPWHGAYSASKFGLRGVSEVLRFDLARHRIGVSVVVPGAVKTGLVQTVEIAGVDREDPDVKKWVDRFAGHAISPEKAAAKILAGVRRNRFLIYTSADIRALYTFKRVAWWPYSVAMRQVNVLFTRALRSGTSPRMLTVVHDTEDANDKASGAGRSLLDEIVRDGARQMLAAALQAEVAAYVAQFADQLDENGHRLVVRNGYHQPREVLTAAGAVQVKAPRVNDRRVDPDTGERKRFSSAILPAWARKSPQMSEVLPLLYLHGLSSNDFTPALEQFLGSGAGLSASTITRLTAQWQDEARAFGARDLSATDYVYLWVDGIHLKVRLDQEKLCLLVMLGVRADGRKELVAITDGYRESAESWADLLRDCKRRGMTAPVLAIGDGALGFWKAVREVFPATKEQRCWFHKQANVLAALPKSAHPSALAAIKEIYNAEDIDKAQIAVKAFEADFGAKYPKAVAKITDDLDVLLEFYKYPAEHWIHLRTTNPIESTFATVRLRTKVTKGPGSRAAGLAMAYKLIDAAAARWRAVNAPHLVALVRAGAVFHKGRLLERPTDITPPTSPSDGGQHAGTEVAMTVWDVVLLVFAGIAGGLTGSIAGLASVATYPALLVVGLPPVAANVTNTVAVVFNGVGSIAGSRPELAGQGAWLKRIIPVAALGGVAGAALLLSTPAEGFEKIVPFLLGFASVAILLPRREHRSARVANHRDHLIRTGVEAAAIFLITIYGGYFGAAAGVLLLALMLRAGGATLPHANAGKNVILGVANLVASAIFVVFAPVYWPAVVPLGIGCLIGSRLGPIIVRHAPSTPLRWLIGVAGIALAIKLALDTYMLQTVAIRGYRSLREVVLPLTELTVITGANGTGKSSVYRALRLLADCGRGQVIGSLAREGGLQSVLWAGPERPSEEAQGATRTRPVSLEMGFAADDFGYVVDLGLPQMAGAPAHRTPSAFTLDPEIKREAVFAGPVLRPSSTLVRRIREFAETAAESGRGFDELSRSLPAYRSVLAEYAHPHALPELSAVSERLRDWRFYDGFRVDAGAPARHPHVGTRTPVLSDDGSDLAAAVQTIIEAGLDDLQRAVADAFDGARVSVAASDGLFDLQLHQRGMLRPLRAAELSDGTLRFLLWAAALLSPSPPSLMVLNEPETSLHPDLVRPLASLIRTAATRTQVVVVTHSRALLEFLDTTPIGDDG'

__HMMER_SH="""
%(ccp4)s/libexec/hmmbuild %(tmpdirname)s/msa.hmm %(tmpdirname)s/msa.fa
%(ccp4)s/libexec/hmmsearch --noali --max -E 1e11 --domE 1e11 --domZ 20600 --tblout %(tmpdirname)s/hmmsearch.log %(tmpdirname)s/msa.hmm  %(db_fname)s
"""
# --max -E 1e10 --domE 1e10
HMMER_SH="""
%(hmmbuild_bin)s %(tmpdirname)s/msa.hmm %(tmpdirname)s/msa.fa
%(hmmsearch_bin)s --noali --max -E 1e11 --domE 1e11 --domZ 20600 --cpu %(cpu)s --tblout %(tmpdirname)s/hmmsearch.log %(tmpdirname)s/msa.hmm  %(db_fname)s
"""

CLUTALW_SH="""
cd %(tmpdirname)s
%(ccp4)s/libexec/clustalw2 input.fa
"""


class model2sequence:


    def __init__(self, mapin=None, mtzin=None, labin=None, badd=None):


        self.nn_model_obj = nn_predict.nn_predict(mapin=mapin, mtzin=mtzin, labin=labin, badd=badd)

        self._xyz_utils = self.nn_model_obj.descr._xyz_utils

        self.ogt = aac.one_letter_given_three_letter
        self.tgo = aac.three_letter_given_one_letter


        # keys: rid = "chainid_resseq" (e.g. "%s_%i" % (res.parent().id, res.resseq_as_int()))
        self.residue_scores_dicts = {}

    #--------------------------------------------------------------------------

    def __seqidentity2reference(self, seq_string, refseq_string):

        ccp4 = os.environ.get('CCP4', None)

        with tempfile.TemporaryDirectory(prefix="guessmysequence_refseq_") as tmpdirname:
            with open(os.path.join(tmpdirname, 'input.fa'), 'w') as ofile:
                ofile.write(">1\n%s\n"%seq_string)
                ofile.write(">2\n%s"%refseq_string)

            clustalw_script = CLUTALW_SH%locals()

            ppipe = subprocess.Popen( clustalw_script,
                                      shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True)

            seq_identity=None
            for stdout_line in iter(ppipe.stdout.readline, ""):
                match = re.search(r'Sequences.*Aligned.*Score:\s*(\d*)', stdout_line)
                if match: seq_identity = int(match.group(1))

            retcode = subprocess.Popen.wait(ppipe)

            return seq_identity


    # -------------------------------------------------------------------------

    def parse_seqdb(self, seq_dbfname, seqid, evalue=None, refseq_string=None):

        seq_string = []
        entry = False
        with gzip.open(seq_dbfname, 'rt') if re.match(r'.*\.gz', seq_dbfname) else open(seq_dbfname, 'rt') as ifile:
            for line in ifile:
                #if re.match(r'^>.*%s .*'%re.escape(seqid), line):
                #if re.match(r'^>%s.*'%re.escape(seqid), line):
                if re.match(r'^>%s[$|\s].*'%re.escape(seqid), line):
                    entry=True
                    continue
                if entry and re.match(r'^>.*', line): break

                if entry: seq_string.append(line.strip())

        if seq_string:
            seq_identity=None
            if refseq_string:
                seq_identity = self.__seqidentity2reference("".join(seq_string), refseq_string)
            seq_identity_str = "" if seq_identity is None else "|sequence_identity=%d"%seq_identity
            evalue_str = "" if evalue is None else "|E-value=%.2e"%evalue
            return ">%s%s" % (seqid, evalue_str)
            # return ">%s%s%s\n%s" % (seqid, evalue_str, seq_identity_str, "\n".join(seq_string))
        else:
            return ""

    # -------------------------------------------------------------------------

    def _parse_hmmer_tblout(self, tblout_fname, nhits=10):
        """
        #                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----

        # target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
        """

        data = []
        if not os.path.isfile(tblout_fname): return data

        with open(tblout_fname, 'r') as ifile:
            for line in ifile:
                if line.startswith("#"): continue

                _entryid = line.split()[0]
                try:
                    # 4 - full sequence E-value
                    # 7 - best 1 domain
                    _eval = float(line.split()[7])
                except:
                    _eval=1.0

                data.append((_entryid, _eval))
                # if len(data)>nhits: break



        return data

    # -------------------------------------------------------------------------

    def query_msa(self, msa_string, db_fname=None, refseq_fname=None, verbose=False, tophits_sto=3):

        msa_seqs=msa_string.split('>')
        seqs=[]
        for line in msa_seqs:
            splitted=line.split('\n')
            if len(splitted)>1:
                seqs.append(pyhmmer.easel.TextSequence(name=bytes(splitted[0],'utf-8'),sequence=splitted[1]))
        
        alphabet = pyhmmer.easel.Alphabet.amino()
        msa=pyhmmer.easel.TextMSA(name=b'msa',sequences=seqs)
        msa_d = msa.digitize(alphabet)
        builder=pyhmmer.plan7.Builder(alphabet)
        background=pyhmmer.plan7.Background(alphabet)
        hmm,_,_=builder.build_msa(msa_d,background)
        
        with pyhmmer.easel.SequenceFile(db_fname,digital=True,alphabet=alphabet) as seq_file:
            cpu = int(os.environ['cpu'])
            hits = next(pyhmmer.hmmer.hmmsearch(hmm,seq_file,cpus=cpu, F1=1e10, F2=1e10, F3=1e10, E=1e10, domE=1e10, incE=1e10, incdomE=1e10))
        
        if len(hits)>0:
            results = []
            for hit in hits:
                results.append((hit.name, str(hit.evalue))) 
            return results
        else:
            return None

    # -------------------------------------------------------------------------

    def model2msa(self, ph,
                        selstr=None,
                        nseqs=100,
                        ofname=None,
                        strictly_protein=False,
                        verbose=False):

        chain_frags = self._xyz_utils.find_chain_fragments(ph, verbose=verbose)

        sel_cache = ph.atom_selection_cache()
        isel = sel_cache.iselection


        ch_resrange = {}
        print( " ==> Chain fragments" )
        for ichf, chf in enumerate(chain_frags):
            ch_resrange[chf[0].parent().id] = [chf[0].resseq_as_int(), chf[-1].resseq_as_int()]
            print( "     %5i %2s %5i:%-5i   [%5i]"  %(ichf+1,
                                                     chf[0].parent().id,
                                                     chf[0].resseq_as_int(),
                                                     chf[-1].resseq_as_int(),
                                                     len(chf)) )


        if selstr is None: return

        # special code used for tests and calibration
        # a selstr in a form "chain X and randm N"
        # selects random chunk of length N in chain X
        match = re.match(r'chain (?P<chain>\w+) and random (?P<random>\d+)', selstr)
        if match:
            _chid = match.group('chain')
            _rnd_len = int(match.group('random'))
            print(" ==> Detected special selection string: random chunk of length %i from chain %s" % (_rnd_len,_chid))
            _resis,_resie = ch_resrange[_chid]
            if _rnd_len>(_resie-_resis):
                print("ERROR: requested fragment length longer than the chain")
                exit(1)

            _rnd_resi = np.random.randint(_resis, _resie-_rnd_len)
            selstr = "chain %s and resi %i:%i" % (_chid, _rnd_resi, _rnd_resi+_rnd_len)

        #print()
        #print( " ==> Selection string: %s" % selstr )
        #print()


        sel_cache = ph.atom_selection_cache()
        isel = sel_cache.iselection
        self.ph_selected = ph.select(isel(selstr))


        chainid='X'
        msa_array = []
        for ch in self.ph_selected.chains():

            #print( " ==> Processing chain %s " % ch.id )

            for conf in ch.conformers():
                #if 'HOH' in [_r.resname.strip() for _r in ch.residues()]: continue

                if strictly_protein and not conf.is_protein(min_content=0.5): continue

                if not conf.altloc in ['', 'A']: continue

                for resi,res in enumerate(conf.residues()[:]):
                    #if resi%10==0: print( "     %5i/%i" % (resi, len(conf.residues())) )

                    scores_dict = self.nn_model_obj.predict(res)
                    if scores_dict is None:
                        if res.resname not in ['HOH', 'DUM', 'WAT']: print("    Incomplete or unknown residue: %s" % res.resname)
                        continue

                    # keep the scores
                    rid = "%s_%i" % (res.parent().parent().id, res.resseq_as_int())
                    self.residue_scores_dicts[rid] = scores_dict

                    _norm = sum([_ for _ in scores_dict.values()])
                    scores_array = [(float(nseqs)*scores_dict[_]/_norm) for _ in self._xyz_utils.standard_aa_names]

                    _thrs=0
                    _dthrs=0.01
                    while True:
                        rounded_array =[np.ceil(_) if _%1.0>_thrs else np.floor(_) for _ in scores_array]
                        if sum(rounded_array)>nseqs:
                            _thrs+=_dthrs
                            if _thrs>1.0: break
                            continue
                        break
                    _aa_array = []
                    for _rn, _num in zip(self._xyz_utils.standard_aa_names, rounded_array):
                        _aa_array.extend( [self.ogt[_rn]]*int(_num) )
                        if verbose: print( "%s:%2i" % (self.ogt[_rn], _num),)
                    if verbose: print( sum(rounded_array) )

                    if len(_aa_array)<nseqs:
                        _aa_array.extend( ['A']*(nseqs-len(_aa_array)) )
                    msa_array.append(_aa_array)

        # MSA columns shuffling...
        #msa_array = np.array(msa_array)
        #for _ in msa_array:
        #    np.random.shuffle(_)
        msa_array = np.array(msa_array).transpose()
        #np.random.shuffle(msa_array)

        if ofname:
            with open(ofname, 'w') as ofile:
                #for _irow, _row in enumerate(np.array(msa_array).transpose().tolist()):
                for _irow, _row in enumerate(msa_array.tolist()):
                    ofile.write( ">%s_%i\n%s\n" % (chainid, _irow, "".join(_row)) )
            print( " ==> Wrote MSA to %s" % ofname )

            return ofname

        msa_string = []
        #for _irow, _row in enumerate(np.array(msa_array).transpose().tolist()):
        for _irow, _row in enumerate(msa_array.tolist()):
            msa_string.append( ">%s_%i\n%s" % (chainid, _irow, "".join(_row)) )
        return "\n".join(msa_string)



    # -------------------------------------------------------------------------


    def calc_alignment_scores(self, frag_scores_array, seq_array):
        scores = []
        for _i in range(len(seq_array)-len(frag_scores_array)+1):
            _s = seq_array[_i:_i+len(frag_scores_array)]
            # there must be exactly as many ones as residues (no unk residues)
            if np.sum(_s)==frag_scores_array.shape[0]:
                scores.append( np.sum(frag_scores_array*_s) )
                continue

            scores.append(1e-6)

        return scores

    # -------------------------------------------------------------------------

    def calc_Gumbel_pvalue(self, alignment_scores, _random_scores_mean, _random_scores_std):
        '''
            estimates Gunmbel distributin p-value for the top-scored alignment in the alignment_scores

            Gumbel correction for normally distributed variables (check Shapiro-Wilk results)
            naively ignoring any correlations:
            Chojnowski and Bochtler Acta Cryst. (2007). A63, 297–305
        '''

        # gets rid of very low scores that may cause numerical issues in Gumbel estimates
        z = np.max([-3.0,(np.max(alignment_scores)-_random_scores_mean)/(_random_scores_std*np.sqrt(2.0))])
        n = max([2.0,float(len(alignment_scores))])
        n = max(1, int(n/10))
        if n>1:
            logn = np.log(n)
            b_n = np.sqrt(2.0*logn)*(1.0 - 0.25*np.log(np.pi*logn)/logn)
            a_n = b_n + 1.0/np.sqrt(2.0*logn)
            #G=1.0-np.exp(-np.exp(a_n*(b_n-z)-0.5*(b_n-z)**2.0))
            #nuerically unstable, check e.g.
            #https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
            G = -np.expm1(-np.exp(a_n*(b_n-z)-0.5*(b_n-z)**2.0))
        else:
            G = erfc(z)

        return G, z, n

    # -------------------------------------------------------------------------

    def query_slide(self, db_fname, refseq_fname=None, verbose=False, tophits_sto=3):

        refseq_string=None
        if refseq_fname:
            refseq = []
            with open(refseq_fname, 'r') as ifile:
                for line in ifile:
                    if line.startswith(">"): continue
                    refseq.append(line.strip())
            refseq_string="".join(refseq)

        rng = np.random.default_rng()

        aa_names_array =  np.array(self._xyz_utils.standard_aa_names)
        frag_scores_array_dict = {}

        # robust, but multiline regexp and very sloooow for large files
        # TODO: implement slide while file is parsed, later...
        #seqdb_objects, non_compliant = any_sequence_format(db_fname)
        seqdb_objects=[]
        db_size=os.path.getsize(db_fname)
        bucket=[]
        with gzip.open(db_fname, 'rt') if re.match(r'.*\.gz', db_fname) else open(db_fname, 'rt') as ifile:
            db_read=0
            for line in ifile:
                if line.startswith('>') and bucket:
                    _obj, _err = any_sequence_format(file_name="wird.fasta", data="".join(bucket))
                    if _obj and not _err and len(set(_obj[0].sequence))>4: seqdb_objects.extend(_obj)
                    bucket=[]

                bucket.append(line)

            _obj, _err = any_sequence_format(file_name="wird.fasta", data="".join(bucket))
            if _obj and not _err:
                seqdb_objects.extend(_obj)

        if not len(seqdb_objects):
            print( " ==> Input database seems to be empty... " )
            return None

        print(" ==> Parsed %i sequeces" % len(seqdb_objects) )

        random_sequence_array = np.array([aa_names_array==self.tgo[_a] for _a in random_sequence], dtype=float)
        random_sequence_array = rng.choice(random_sequence_array, 1000000, replace=True, axis=0)

        # query DB with (up to) two longest chain frags
        chain_frags = self._xyz_utils.find_chain_fragments(self.ph_selected, verbose=verbose)
        selected_chain_frags = chain_frags[:1]
        print(" ==> Making a query with the longest chain fragment%s: %s" % ("s" if len(chain_frags)>1 else "",
                                                               ",".join(tuple(map(str,map(len,selected_chain_frags))))
                                                               ))
        # estimate background distribution params (once!)
        for ichf, chf in enumerate(selected_chain_frags):
            frag_scores = []
            for res in chf:
                rid = "%s_%i" % (res.parent().id, res.resseq_as_int())
                _score = self.residue_scores_dicts.get(rid, None)
                if _score is None:
                    print("     Incomplete or unknown residue: %s" % rid)
                    continue

                frag_scores.append( _score  )

            frag_scores_array = np.array([[np.log(np.clip(_s[_a],1e-33,1.0)) for _a in aa_names_array] for _s in frag_scores], dtype=float)
            base_scores = self.calc_alignment_scores(frag_scores_array, random_sequence_array)
            _random_scores_mean,_random_scores_std = np.mean(base_scores), np.std(base_scores)

            # Shapiro-Wilk normality test for the background distribution
            # Gumble correction cannot be used if not passed!
            if verbose:
                shapiro_test = stats.shapiro(base_scores[:5000])
                print(" ==> Background distribution Shapiro-Wilk normality test p-value %.4e"%shapiro_test.pvalue)

            frag_scores_array_dict[rid] = (frag_scores_array, _random_scores_mean,_random_scores_std)


        results = []
        for _seq in seqdb_objects[:]:
            #_cleaned_sequence = _seq.sequence.replace('X','')
            _cleaned_sequence = _seq.sequence

            try:
                target_sequence_array = np.array([aa_names_array==self.tgo[_a] for _a in _seq.sequence], dtype=float)
            except:
                continue

            # allow for only a few non-standard residues
            #if sum(np.max(target_sequence_array, axis=1))<0.95*len(_cleaned_sequence): continue

            for _frag_id, _frag_data in frag_scores_array_dict.items():
                _frag_array, _random_scores_mean,_random_scores_std = _frag_data
                alignment_scores = self.calc_alignment_scores(_frag_array, target_sequence_array)

                if not alignment_scores:
                    continue


                best_match_index = np.argmax(alignment_scores)

                ##z = (np.max(alignment_scores)-_random_scores_mean)/(_random_scores_std*np.sqrt(2.0))
                ## gets rid of crap that is useless, but may cause numerical issies in Gumbel
                #z = np.max([-3.0,(np.max(alignment_scores)-_random_scores_mean)/(_random_scores_std*np.sqrt(2.0))])
                #n = max([2.0,float(len(alignment_scores))])
                #n = max(1, int(n/10))
                #if n>1:#: and shapiro_test.pvalue<1e-2:
                #    # Gumbel correction for normally distributed variables (check Shapiro-Wilk results)
                #    # naivley ignores any correlations
                #    # Chojnowski and Bochtler Acta Cryst. (2007). A63, 297–305
                #    logn = np.log(n)
                #    b_n = np.sqrt(2.0*logn)*(1.0 - 0.25*np.log(np.math.pi*logn)/logn)
                #    a_n = b_n + 1.0/np.sqrt(2.0*logn)
                #    #G=1.0-np.exp(-np.exp(a_n*(b_n-z)-0.5*(b_n-z)**2.0))
                #    #https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
                #    pvalue = -np.expm1(-np.exp(a_n*(b_n-z)-0.5*(b_n-z)**2.0))
                #else:
                #    pvalue = erfc(z)

                pvalue, z, n = self.calc_Gumbel_pvalue(alignment_scores, _random_scores_mean, _random_scores_std)
                results.append( [_seq.name, pvalue, erfc(z), n, _cleaned_sequence] )

        if not results: return None
        # results are stored for all the fargs and may be repeated
        formatted_results=[]
        for _res in sorted(results, key=lambda _:_[1])[:tophits_sto]:
            _s_name, _s_pv, _s_erfc, _s_n, _s_str = _res

            if refseq_string:
                seq_identity = self.__seqidentity2reference( _s_str, refseq_string)
                if seq_identity is None:
                    seqid_str = "|sequence_identity=ERROR"
                else:
                    seqid_str = "|sequence_identity=%d"%seq_identity
            else:
                seqid_str = ""

            if verbose:
                fasta_str = ">%s|n=%d|erfc=%.2e|p-value=%.2e%s\n%s"%(_s_name, _s_n, _s_erfc, _s_pv, seqid_str, _s_str)
            else:
                fasta_str = ">%s|p-value=%.2e%s\n%s"%(_s_name, _s_pv, seqid_str, _s_str)
            # (seq_id, E-value, sequence with e-value in a title line)
            formatted_results.append( (_s_name, _s_pv, fasta_str) )

        return formatted_results


    # -------------------------------------------------------------------------

    def align_frags(self, target_sequence,
                          modelout,
                          verbose=False):


        # do some VERY CRUDE cleaning, in case it's a FASTA string with a header
        target_sequence_processed = []
        for line in target_sequence.split('\n'):
            if line.startswith('>'): continue
            target_sequence_processed.append(line)

        target_sequence_processed = "".join(target_sequence_processed)

        aa_names_array =  np.array(self._xyz_utils.standard_aa_names)
        target_sequence_array = np.array([aa_names_array==self.tgo[_a] for _a in target_sequence_processed], dtype=float)
        rng = np.random.default_rng()
        rng = np.random.default_rng(seed=123)
        random_sequence_array = rng.choice(target_sequence_array, 10000, replace=True, axis=0)

        chain_frags = self._xyz_utils.find_chain_fragments(self.ph_selected, verbose=verbose)

        for ichf, chf in enumerate(chain_frags):

            frag_scores = []
            for res in chf:
                rid = "%s_%i" % (res.parent().id, res.resseq_as_int())
                _score = self.residue_scores_dicts.get(rid, None)
                assert _score is not None
                frag_scores.append( _score  )


            frag_scores_array = np.array([[np.log(np.clip(_s[_a],1e-33,1.0)) for _a in aa_names_array] for _s in frag_scores], dtype=float)
            alignment_scores = self.calc_alignment_scores(frag_scores_array, target_sequence_array)

            base_scores = self.calc_alignment_scores(frag_scores_array, random_sequence_array)
            _random_scores_mean,_random_scores_std = np.mean(base_scores), np.std(base_scores)

            # Added:
            # sovling the argmax of empty array error
            if not alignment_scores:
                print(" ==> Empty alignment scores returned")
                continue
            
            best_match_index = np.argmax(alignment_scores)
            #n = max([2.0,float(len(alignment_scores))])
            #z = np.max([-3.0,(np.max(alignment_scores)-_random_scores_mean)/(_random_scores_std*np.sqrt(2.0))])
            #n = max(1, int(n/10))
            #if n>1:
            #    logn = np.log(n)
            #    b_n = np.sqrt(2.0*logn)*(1.0 - 0.25*np.log(np.math.pi*logn)/logn)
            #    a_n = b_n + 1.0/np.sqrt(2.0*logn)
            #    pvalue = -np.expm1(-np.exp(a_n*(b_n-z)-0.5*(b_n-z)**2.0))
            #else:
            #    pvalue = erfc(z)

            pvalue, z, n = self.calc_Gumbel_pvalue(alignment_scores, _random_scores_mean, _random_scores_std)

            if verbose: print("DEBUG_STATS_n_pv_lnpv_erfc", n, pvalue, "%e"%(-np.log(pvalue)), "%e"%erfc( (np.max(alignment_scores)-_random_scores_mean)/(_random_scores_std*np.sqrt(2.0)) ))

            if pvalue>999:
                print(" ==> No reliable sequence alignment found")
                continue

            print( "     %5i %2s %5i:%-5i @ %5i:%-5i p-value=%e"  %(ichf+1,
                                                                      chf[0].parent().id,
                                                                      chf[0].resseq_as_int(),
                                                                      chf[-1].resseq_as_int(),
                                                                      best_match_index+1,
                                                                      best_match_index+len(chf),
                                                                      pvalue))#-np.log(pvalue)/np.log(10)) )



            target_seq_segment = target_sequence_processed[best_match_index:best_match_index+len(chf)]
            print( "           ", "".join([self.ogt.get(_r.only_atom_group().resname, "?") for _r in chf]) )
            print( "           ", "".join(target_seq_segment) )

            rscc_array = []
            for res, new_resname, new_resseq in zip(chf,
                                                    target_seq_segment,
                                                    range(best_match_index+1,best_match_index+len(chf)+1)):
                _rscc = self._xyz_utils.mutate_aa(res, target_aa=self.tgo[new_resname], target_resseq=new_resseq)
                try:
                    rscc_array.append(str(min(9,int(10*_rscc))))
                except:
                    rscc_array.append("-")

            print( "           ", "".join(rscc_array) )
            print()

        if modelout:
            self.ph_selected.write_pdb_file(modelout)
            print( " ==> Wrote modified model to:\n      %s" % modelout)