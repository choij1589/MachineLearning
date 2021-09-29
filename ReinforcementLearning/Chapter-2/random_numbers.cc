// g++ -L `root-config --cflags --glibs`
#include <iostream>
#include <string>
#include <random>
#include <TCanvas.h>
#include <TH1D.h>

int main() {
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<> die(1, 6);
	std::uniform_real_distribution<> uniform(0., 1.);
	std::normal_distribution<double> gaus(0., 1.);
	
	TCanvas* c = new TCanvas("c", "");
	TH1D* h_die = new TH1D("h_die", "", 6, 1., 7.);
	TH1D* h_uni = new TH1D("h_uni", "", 100, 0., 1.);
	TH1D* h_gaus = new TH1D("h_gaus", "", 100, -5, 5);

	for (int i = 0; i < 100000; i++) {
		h_die->Fill(die(generator));
		h_uni->Fill(uniform(generator));
		h_gaus->Fill(gaus(generator));
	}

	c->cd();
	h_die->Draw();
	c->SaveAs("random/die.png");

	h_uni->Draw();
	c->SaveAs("random/uniform.png");

	h_gaus->Draw();
	c->SaveAs("random/gauss.png");

	delete c, h_die, h_uni, h_gaus;

	return 0;
}
