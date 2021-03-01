# -------------------------Import------------------------
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import norm


# ------------------------Class------------------------
class Coats:
    """ COATS (CO2 Abatement Tied to Surface engineering) is a prospective tool to
        calculate envieonmental burdens and benefits [GHG, EQ, HH] of applying novel
        surface engineering technologies to the energy sector.
    """

    # Initializer / Instance attributes
    def __init__(self, d_elec, imp_elec, coat_impacts, coat_elec, elec_char, adop_rates,
                 year_list, scenario_list, electricity_tech_list, electricity_tech_app_se_list,
                 electricity_tech_se_list, regions_list, impact_list):
        """Initialie all data needed
         Args:
        -----
            d_elec: [df] Market share of electricity technology in EJ for each scenario, region and electricity technology
            imp_elec: [df] Impacts of electricity technologies [CC,EQ,HH] per kWh electricity supplied
            coat_impacts: [df] Impacts (excluding electricity) of coating technologies [CC,EQ,HH] per kWh electricity supplied
            coat_elec: [df] Electricity consumption (in kWh) of coating technologies per kWh electricity supplied
            elec_char: [df] Electricity technology characteristics (lifetime and efficiency improvement from SE)
            adop_rates: [df] Adoption rates for old and new cohort of plants for different SSPs
            year_list: [Serie] Years of calculation
            scenario_list: [Serie] Name of the used SSP scenarios
            electricity_tech_list: [Serie] Name of all electricity conversion technologies in MESSAGE models
            electricity_tech_app_se_list: [Serie] Name of electricity conversion technologies where SE is applicable
            electricity_tech_se_list: [Serie] Name of electricity conversion technologies where SE is applied
            regions_list: [Serie] Name of all regions
            impact_list: [Serie] Name of all impact categories
        """

        # Initial atributes
        self.d_elec = d_elec
        self.imp_elec = imp_elec
        self.coat_impacts = coat_impacts
        self.coat_elec = coat_elec
        self.elec_char = elec_char
        self.adop_rates = adop_rates
        self.year_list = year_list
        self.scenario_list = scenario_list
        self.electricity_tech_list = electricity_tech_list
        self.electricity_tech_app_se_list = electricity_tech_app_se_list
        self.electricity_tech_se_list = electricity_tech_se_list
        self.regions_list = regions_list
        self.impact_list = impact_list

        # To calculate
        self.d_elec_se = pd.DataFrame()
        self.energy_results = pd.DataFrame()
        self.energy_total_region = pd.DataFrame()
        self.imp_elec_diag = pd.DataFrame()
        self.d_elec_results = pd.DataFrame()
        self.d_elec_results_se = pd.DataFrame()
        self.d_impact_coating_el = pd.DataFrame()
        self.d_impact_coating_pr = pd.DataFrame()
        self.d_coating_impact = pd.DataFrame()
        self.d_elec_results_total = pd.DataFrame()
        self.benefits_from_se = pd.DataFrame()
        self.burdens_from_se = pd.DataFrame()
        self.burdens_from_se_tech = pd.DataFrame()

    @classmethod
    def from_template(cls, template_name):
        """ Import all data needed for the calculation from the Excel sheet.
        Args:
        -----
            template_name: [string] the name of the excel file to read (iamc_db.xlsx)
        Returns:
        --------
            scenario_object:
        """
        # read excel
        d_elec = pd.read_excel(template_name, index_col=[0, 2, 1], sheet_name='data').fillna(0.0)
        imp_elec = pd.read_excel(template_name, index_col=[0, 1], sheet_name='impact_elec').fillna(0.0)
        coat_impacts = pd.read_excel(template_name, index_col=[0, 1], sheet_name='impact_coat').fillna(0.0)
        coat_elec = pd.read_excel(template_name, index_col=[0], sheet_name='elec_coat').fillna(0.0)
        elec_char = pd.read_excel(template_name, index_col=0, sheet_name='elec_char').fillna(0.0)
        adop_rates = pd.read_excel(template_name, index_col=0, sheet_name='scenarios').fillna(0.0)

        # lists used to build index
        year_list = [i for i in range(2020, 2110, 10)]
        scenario_list = list(d_elec.index.get_level_values(0).unique())
        electricity_tech_list = list(d_elec.index.get_level_values(1).unique())
        regions_list = list(d_elec.index.get_level_values(2).unique())
        impact_list = list(imp_elec.index.get_level_values(1).unique())
        electricity_tech_app_se_list = list(elec_char.index.get_level_values(0).unique())
        electricity_tech_se_list = list(coat_impacts.index.get_level_values(0).unique())

        # create object
        scenario_object = cls(d_elec, imp_elec, coat_impacts, coat_elec, elec_char, adop_rates,
                              year_list, scenario_list, electricity_tech_list, electricity_tech_app_se_list,
                              electricity_tech_se_list, regions_list, impact_list)
        return scenario_object

    def solve(self):
        """Solve all methods
        """
        self._calc_energy_results()
        self._calc_energy_total_region()
        self._calc_d_elec_results()
        self._calc_d_coating_impact()
        self._calc_benefits_from_se()
        self._calc_burdens_from_se()

    def _calc_energy_results(self):
        """ Calculation of the market share of electricity technologies where SE is applied
        Args:
        -----
        Returns:
        --------
            self.energy_results: [df] Market share of electricity technology including SE-enhanced technologies
        """
        # create a new index to show the share of energy supplied by technologies with SE with respect to the total energy supplied
        new_index = self.electricity_tech_se_list.copy()
        new_index.insert(len(self.electricity_tech_se_list), 'share of total production')
        self.energy_results = pd.DataFrame(index=pd.MultiIndex.from_product([self.scenario_list, new_index]),
                                           columns=self.year_list)
        self.energy_results = self.energy_results.fillna(0.0)
        self.d_elec_se = self.d_elec.copy()

        # track the vintage of power plants where SE is applied for all the scenarios to know the energy supplied by SE-enhanced technologies
        for scenario in self.scenario_list:
            self.__calc_vintage_track_scenario(scenario)
            for electricity_tech_se in self.electricity_tech_se_list:
                self.energy_results.loc[(scenario, electricity_tech_se)].at[self.year_list] = list(
                    self.d_elec_se.loc[scenario, electricity_tech_se][self.year_list].sum())
            self.energy_results.loc[(scenario, 'share of total production')].at[self.year_list] = (
                                                                                                      self.energy_results.loc[
                                                                                                          scenario].sum()) / (
                                                                                                      self.d_elec_se.loc[
                                                                                                          scenario].sum())

    def __calc_vintage_track_region(self, scenario, technology, region, step=10, base_year=2010):
        """ Calculation of the market share of the electricity technologies adopting SE
        Args:
        -----
            scenario: [string] Name of the scenario
            technology: [string] Name of the electricity technology
            region: [string] Name of the region
            step: [integer] the step (in years) of time in the MESSAGE model
            base_year: [integer] the initial year for the vintage tracking where the normal distribution is applied
        Returns:
        --------
            self.d_elec_se: [df] Market share of electricity technology including SE-enhanced technologies
        """

        # add the base year at the beginning to estimate the energy supplied by each age cohort
        in_years = self.year_list.copy()
        in_years.insert(0, base_year)

        # vintage is the age cohort table for the electricity technology where the SE is applied
        vintage = pd.DataFrame(np.full((self.elec_char.loc[technology, 'Age'] // step, len(self.year_list) + 1), 0.0),
                               index=pd.Series(range(0, self.elec_char.loc[technology, 'Age'] // step)),
                               columns=in_years)

        # Fill base_year according to the normal distributions
        mu = (step + self.elec_char.loc[technology, 'Age']) / 2  # mean of the normal distribution
        std = step  # standard deviation of the normal distribution
        x = np.linspace(step, self.elec_char.loc[technology, 'Age'], self.elec_char.loc[technology, 'Age'] // step)
        pdf = norm.pdf(x, mu, std)  # probability distribution function of the normal distribution
        vintage.at[:, base_year] = (pdf * self.d_elec.loc[(scenario, technology)].loc[region, base_year]) / pdf.sum()

        # Fill all the values in the age cohort table
        for year in self.year_list:
            for agecohort in range(0, self.elec_char.loc[technology, 'Age'] // step):
                if agecohort == 0:
                    vintage.at[agecohort, year] = self.d_elec.loc[(scenario, technology)].loc[region, year] - \
                                                  self.d_elec.loc[(scenario, technology)].loc[region, year - step] + \
                                                  vintage.loc[
                                                      (self.elec_char.loc[technology, 'Age'] // step) - 1, year - step]
                else:
                    vintage.at[agecohort, year] = vintage.loc[agecohort - 1, year - step]

            # Get rid of negatives through early decommisioning
            if vintage.loc[0, year] < 0:
                old_tech_index = 0
                while old_tech_index < (self.elec_char.loc[technology, 'Age'] // step) - 1 and vintage.loc[0, year] < 0:
                    early_decom_cap = vintage.loc[0, year]
                    poss_decom_cap = vintage.loc[
                        self.elec_char.loc[technology, 'Age'] // step - 1 - old_tech_index, year]
                    if poss_decom_cap >= abs(early_decom_cap):
                        vintage.at[self.elec_char.loc[
                                       technology, 'Age'] // step - 1 - old_tech_index, year] = poss_decom_cap + early_decom_cap
                        vintage.at[0, year] = 0
                    else:
                        vintage.at[self.elec_char.loc[technology, 'Age'] // step - 1 - old_tech_index, year] = 0
                        vintage.at[0, year] = early_decom_cap + poss_decom_cap
                    old_tech_index += 1

        # adoptions rates
        # additional restriction for wind because a fraction of the energy is supplied in the winter and a fraction of the countries are cold
        cold_OECD_regions = 0.06
        cold_REF_regions = 1
        wind_supplied_winter = 0.5
        if technology == 'Secondary Energy|Electricity|Wind' and region == 'R5.2OECD':
            adoption_old = cold_OECD_regions * wind_supplied_winter * self.adop_rates.loc['old', scenario]
            adoption_new = cold_OECD_regions * wind_supplied_winter * self.adop_rates.loc['new', scenario]
        elif technology == 'Secondary Energy|Electricity|Wind' and region == 'R5.2REF':
            adoption_old = cold_REF_regions * wind_supplied_winter * self.adop_rates.loc['old', scenario]
            adoption_new = cold_REF_regions * wind_supplied_winter * self.adop_rates.loc['new', scenario]
        else:
            adoption_old = self.adop_rates.loc['old', scenario]
            adoption_new = self.adop_rates.loc['new', scenario]

        # create triangle matrix with the adoption rates of SE in old and new plants
        tri_old_no_se = pd.DataFrame(
            np.triu(np.full((len(self.year_list), self.elec_char.loc[technology, 'Age'] // step), 1 - adoption_old), 0),
            index=self.year_list,
            columns=vintage.index)
        tri_old_se = pd.DataFrame(
            np.triu(np.full((len(self.year_list), self.elec_char.loc[technology, 'Age'] // step), adoption_old), 0),
            index=self.year_list,
            columns=vintage.index)
        tri_new_no_se = pd.DataFrame(
            np.tril(np.full((len(self.year_list), self.elec_char.loc[technology, 'Age'] // step), 1 - adoption_new),
                    -1),
            index=self.year_list,
            columns=vintage.index)
        tri_new_se = pd.DataFrame(
            np.tril(np.full((len(self.year_list), self.elec_char.loc[technology, 'Age'] // step), adoption_new), -1),
            index=self.year_list,
            columns=vintage.index)

        # calculate the energy supplied by electricity technologies without SE
        vintage_old_no_se = (tri_old_no_se.T * vintage.loc[:, self.year_list]).sum()
        vintage_new_no_se = (tri_new_no_se.T * vintage.loc[:, self.year_list]).sum()
        self.d_elec_se.loc[(scenario, technology)].at[region, self.year_list] = np.add(vintage_old_no_se,
                                                                                       vintage_new_no_se)

        # calculate the energy supplied by electricity technologies with SE
        vintage_old_se = (tri_old_se.T * vintage.loc[:, self.year_list]).sum()
        vintage_new_se = (tri_new_se.T * vintage.loc[:, self.year_list]).sum()
        self.d_elec_se.loc[(scenario, technology + '|w/ SE')].at[region, self.year_list] = np.add(vintage_old_se,
                                                                                                  vintage_new_se)

    def __calc_vintage_track_tech(self, scenario, technology):
        """ Calculation of the market share of the electricity technologies adopting SE per technology
        Args:
        -----
            scenario: [string] Name of the scenario
            technology: [string] Name of the electricity technology
        Returns:
        --------
        """
        # For wind, SE is only applied to two regions which are considered "cold" regions
        if technology == 'Secondary Energy|Electricity|Wind':
            self.__calc_vintage_track_region(scenario, technology, 'R5.2OECD')
            self.__calc_vintage_track_region(scenario, technology, 'R5.2REF')
        else:
            for region in self.regions_list:
                self.__calc_vintage_track_region(scenario, technology, region)

    def __calc_vintage_track_scenario(self, scenario):
        """ Calculation of the market share of the electricity technologies adopting SE per scenario
        Args:
        -----
            scenario: [string] Name of the scenario
        Returns:
        --------
        """
        for electricity_tech_se in self.electricity_tech_app_se_list:
            self.__calc_vintage_track_tech(scenario, electricity_tech_se)

    def _calc_energy_total_region(self):
        """ Calculation of the market share of the electricity technologies adopting SE per region
        Args:
        -----
        Returns:
        --------
        """
        self.energy_total_region = pd.DataFrame(
            np.full((len(self.scenario_list) * len(self.regions_list), len(self.year_list)), 0.0),
            index=pd.MultiIndex.from_product([self.scenario_list, self.regions_list]),
            columns=self.year_list)
        for scenario in self.scenario_list:
            for region in self.regions_list:
                variable = self.d_elec.loc[scenario, self.year_list].copy()
                variable = variable.swaplevel(0, 1, axis=0)
                variable = pd.DataFrame(variable.sort_values(by=['Region'], kind='mergesort').loc[region].sum(axis=0))
                variable = variable[0].to_list()
                self.energy_total_region.loc[(scenario, region)].at[self.year_list] = variable

    def _calc_d_elec_results(self):
        """ Calculate the environmental impact from the original SSPs (without SE)
        Args:
        -----
        Returns:
        --------
            self.d_elec_results: [df] The environmental impact from the original SSPs
            self.d_elec_results_se: [df] The environmental impact from the SSPs with SE
        """

        # fill in the impact of SE technologies
        eff = 1 - self.elec_char.loc[:, 'EffieciencyImprovement']
        for tech in self.electricity_tech_app_se_list:
            tech_se = tech + '|w/ SE'
            imp_elec_se = self.imp_elec.loc[tech] * eff.loc[tech]
            imp_elec_se.index = pd.MultiIndex.from_product([[tech_se], imp_elec_se.index])
            self.imp_elec.loc[tech_se] = imp_elec_se

        # create empty sets for the output data
        self.d_elec_results = []
        self.d_elec_results_se = []
        list1 = [self.imp_elec.loc[i] for i in self.electricity_tech_list]
        for scenario in self.scenario_list:
            self.imp_elec_diag = pd.DataFrame(linalg.block_diag(*list1),
                                              index=pd.MultiIndex.from_product(
                                                  [self.electricity_tech_list, self.impact_list]),
                                              columns=self.d_elec.loc[scenario].index)

            d_elec_results_temp = self.imp_elec_diag @ (self.d_elec.loc[scenario]).div(3.6)
            self.d_elec_results.append(d_elec_results_temp)
            d_elec_results_se_temp = self.imp_elec_diag @ (self.d_elec_se.loc[scenario]).div(3.6)
            self.d_elec_results_se.append(d_elec_results_se_temp)
        self.d_elec_results = pd.concat(self.d_elec_results)
        self.d_elec_results_se = pd.concat(self.d_elec_results_se)

        # add scenario name at the beginning
        self.d_elec_results = self.d_elec_results.set_index(
            pd.MultiIndex.from_product([self.scenario_list, self.electricity_tech_list, self.impact_list]))
        self.d_elec_results_se = self.d_elec_results_se.set_index(
            pd.MultiIndex.from_product([self.scenario_list, self.electricity_tech_list, self.impact_list]))

    def _calc_d_coating_impact(self):
        """ Calculate the impact of the coating process
        Args:
        -----
        Returns:
        --------
            self.d_impact_coating_el: [df] The impact from the electricity used in the coating process
            self.d_impact_coating_pr: [df] The impact from the coating process (exluding electricity)
            self.d_coating_impact: [df] The total impact of the coating process
            self.d_elec_results_total: [df] The net benefits of applying surface engineering
        """
        # this gets the total electricity needed for coating per region and technology
        d_elec_region = self.d_elec_se.copy()
        d_elec_region = d_elec_region.loc[pd.IndexSlice[:, self.electricity_tech_se_list, :], :]
        d_elec_region = d_elec_region.reorder_levels([0, 2, 1], axis=0).sort_values(by=['Region'], kind='mergesort')

        coat_elec_regions = pd.DataFrame(self.coat_elec['electricity (kwh/kwh)'].to_list() * 5,
                                         index=pd.MultiIndex.from_product([self.regions_list, self.coat_elec.index]))

        imp_coat_elec = pd.DataFrame(np.diag(coat_elec_regions[0].to_list()),
                                     index=pd.MultiIndex.from_product([self.regions_list, self.coat_elec.index]),
                                     columns=pd.MultiIndex.from_product([self.regions_list, self.coat_elec.index]))

        self.d_impact_coating_el = []
        self.d_impact_coating_pr = []
        self.d_coating_impact = []
        for scenario in self.scenario_list:
            elec_needed = imp_coat_elec @ (d_elec_region.loc[scenario]).div(3.6)

            imp_elec_diag_2 = self.imp_elec_diag.loc[
                pd.IndexSlice[self.electricity_tech_se_list], self.electricity_tech_se_list]
            imp_elec_diag_2 = imp_elec_diag_2.reorder_levels([1, 0], axis=1).sort_values(by=['Region'],
                                                                                         kind='mergesort', axis=1)

            impact_coating_el = imp_elec_diag_2 @ elec_needed

            # this gets the impact of coating
            coat_impact_diag = pd.DataFrame(np.diag(list(self.coat_impacts.loc[:, 'Impact / kWh'])),
                                            index=pd.MultiIndex.from_product(
                                                [self.electricity_tech_se_list, self.impact_list]),
                                            columns=pd.MultiIndex.from_product(
                                                [self.electricity_tech_se_list, self.impact_list]))

            d_elec_se_sum = self.d_elec_se.copy()
            d_elec_se_sum = (
            (d_elec_se_sum.loc[pd.IndexSlice[:, self.electricity_tech_se_list, :], self.year_list]).loc[scenario]).sum(
                level=0, axis=0)
            d_elec_se_sum = pd.concat([d_elec_se_sum] * len(self.impact_list))
            d_elec_se_sum = d_elec_se_sum.sort_values(by=['Variable'])
            impact_list_extended = self.impact_list * len(self.electricity_tech_se_list)
            d_elec_se_sum['impact'] = impact_list_extended
            d_elec_se_sum = d_elec_se_sum.set_index([d_elec_se_sum.index, 'impact'])

            impact_coating_pr = coat_impact_diag @ d_elec_se_sum

            impact_coating_el_sum = impact_coating_el.groupby(level=1).sum()
            impact_coating_pr_sum = impact_coating_pr.groupby(level=1).sum()
            total_impact_coating = impact_coating_el_sum.add(impact_coating_pr_sum, fill_value=0.0)

            benefits_results = self.d_elec_results.loc[scenario].groupby(level=[1]).sum() - self.d_elec_results_se.loc[
                scenario].groupby(level=[1]).sum()
            d_net_results = benefits_results - total_impact_coating

            self.d_coating_impact.append(total_impact_coating)
            self.d_impact_coating_el.append(impact_coating_el)
            self.d_impact_coating_pr.append(impact_coating_pr)
        self.d_coating_impact = pd.concat(self.d_coating_impact)
        self.d_impact_coating_el = pd.concat(self.d_impact_coating_el)
        self.d_impact_coating_pr = pd.concat(self.d_impact_coating_pr)

        # get the results ready for graph 2
        graph2 = []
        graph2_se = []
        for scenario in self.scenario_list:
            d_elec_results_impacts = self.d_elec_results.loc[scenario].groupby(level=[1]).sum()
            d_elec_results_impacts_se = self.d_elec_results_se.loc[scenario].groupby(level=[1]).sum()
            graph2.append(d_elec_results_impacts)
            graph2_se.append(d_elec_results_impacts_se)
        d_elec_results_impacts = pd.concat(graph2)
        d_elec_results_impacts_se = pd.concat(graph2_se)

        d_elec_results_impacts = d_elec_results_impacts.reset_index()
        d_elec_results_impacts = d_elec_results_impacts.drop(['index'], axis=1)

        # add the impact of coating
        d_elec_results_impacts_se = d_elec_results_impacts_se.add(self.d_coating_impact)
        d_elec_results_impacts_se = d_elec_results_impacts_se.reset_index()
        d_elec_results_impacts_se = d_elec_results_impacts_se.drop(['index'], axis=1)

        self.d_elec_results_total = pd.concat([d_elec_results_impacts, d_elec_results_impacts_se])
        self.d_elec_results_total = self.d_elec_results_total.set_index(
            pd.MultiIndex.from_product([['noSE', 'SE'], self.scenario_list, self.impact_list]))
        self.d_elec_results_total = self.d_elec_results_total.swaplevel(0, 2, axis=0)
        self.d_elec_results_total = self.d_elec_results_total.sort_index(ascending=True)

    def _calc_benefits_from_se(self):
        """ Calculate the benefits from surface engineering (in terms of tCO2-eq reduction potential)
        Args:
        -----
        Returns:
        --------
            self.benefits_from_se: [df] The benefits from surface engineering
        """
        self.benefits_from_se = self.d_elec_results - self.d_elec_results_se
        self.benefits_from_se = self.benefits_from_se.swaplevel(0, 2, axis=0)
        self.benefits_from_se = self.benefits_from_se.swaplevel(1, 2, axis=0)
        self.benefits_from_se = self.benefits_from_se.loc['CC']
        self.benefits_from_se = self.benefits_from_se.loc[['SSP2-Baseline', 'SSP1-Baseline', 'SSP3-Baseline']]

    def _calc_burdens_from_se(self):
        """ Calculate the impacts of the surface engineering process
        Args:
        -----
        Returns:
        --------
            self.burdens_from_se: [df] The impacts from surface engineering
        """
        self.burdens_from_se = self.d_impact_coating_el.loc[:, self.year_list] + self.d_impact_coating_pr
        self.burdens_from_se = self.burdens_from_se.set_index(
            pd.MultiIndex.from_product([self.scenario_list, self.electricity_tech_se_list, self.impact_list]))
        self.burdens_from_se = self.burdens_from_se.swaplevel(0, 2, axis=0)
        self.burdens_from_se = self.burdens_from_se.swaplevel(1, 2, axis=0)
        self.burdens_from_se = self.burdens_from_se.loc['CC']
        self.burdens_from_se = self.burdens_from_se.loc[['SSP2-Baseline', 'SSP1-Baseline', 'SSP3-Baseline']]
        self.burdens_from_se_tech = self.burdens_from_se.copy()
        self.burdens_from_se = self.burdens_from_se.groupby(level=0).sum()