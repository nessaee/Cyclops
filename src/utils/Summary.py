from .Markov import Markov
from .config import *
from .Helper import hamming, number_sequence, similarity
from .Sequence import Sequences
from .Adversary import Adversary
from .Protocol import Protocol
class Summary:
    def __init__(
            self, 
            protocol : Protocol,
            histogram=False, 
            statistics=False,    
        ):
        self.protocol = protocol
        if statistics: self.stats()
        if histogram:  self.target_histogram(show=True)

    def stats(self):
        v = pd.Series(number_sequence(self.v_sequence))
        c = pd.Series(number_sequence(self.c_sequence))
        self.c_statistics = c.describe()
        self.v_statistics = v.describe()

    def table(self,ri=False,exp=True,counts=True):
        print("Ri:", ri)
        cpr_avg = self.protocol.avg_cpr_ri if ri else self.protocol.cpr_r
        apr_avg = self.protocol.avg_apr_ri if ri else self.protocol.apr_r

        cpr_sd = self.protocol.sd_cpr_ri if ri else self.protocol.cpr_r
        apr_sd = self.protocol.sd_apr_ri if ri else self.protocol.apr_r

        cpr_avg = round(cpr_avg*100,4)
        apr_avg = round(apr_avg*100,4)
        cpr_sd = round(cpr_sd*100,4)
        apr_sd = round(apr_sd*100,4)
        
        cexp = self.protocol.cscores
        aexp = self.protocol.ascores
        #a_counts = [self.protocol.a_total_test_count, self.protocol.a_valid_test_count]
        #c_counts = [self.protocol.c_total_test_count, self.protocol.c_valid_test_count]
        #print(f"ADVERSARY TEST COUNTS: {a_counts} ")
        #print(f"CANDIDATE TEST COUNTS: {c_counts} ")
        ret = [cpr_avg, apr_avg, cexp, aexp, cpr_sd, apr_sd] if exp else [cpr_avg,apr_avg]
        return ret# + [self.protocol.c_valid_test_count, self.protocol.a_valid_test_count] if counts else ret

    
    def target_histogram(self, show=False):
        v_num_targets = self.number_sequence(self.v_sequence)
        c_num_targets = self.number_sequence(self.c_sequence)
        num_bins = max(max(set(v_num_targets)), max(set(c_num_targets)))
        bins = [i for i in range(num_bins+2)]

        vcounts, vbins = np.histogram(v_num_targets, bins=bins)
        self.vfrequency = {vbins[i]:vcounts[i]/sum(vcounts) for i in range(len(vcounts))}

        ccounts, cbins = np.histogram(c_num_targets, bins=bins)
        self.cfrequency = {cbins[i]:ccounts[i]/sum(ccounts) for i in range(len(ccounts))}

        if show:
            self.show_state_frequencies()
            plt.stairs(vcounts, vbins, label='Verifier')
            plt.stairs(ccounts, cbins, label='Candidate')

            plt.title('Target Quantity Histogram')
            plt.xlabel('Number of targets')
            plt.ylabel('Frequency')
            plt.legend(prop={'size': 12})
            plt.xticks(bins)

            plt.show()
            plt.clf()
    
    def show_state_frequencies(self):
        from tabulate import tabulate
        table = [["State", "Pr(State)"]]
        for state in self.vfrequency:
            table.append([state, self.vfrequency[state]])
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    
