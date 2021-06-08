typedef vector<vector<float>> f_matrix
typedef vector<float> f_vector

class Gmm {
private:
    f_matrix dataset;
    int n_clusters, x_dim;

public:
    Gmm (f_matrix dataset, int x_dim) {
        dataset = dataset;

    }

    f_matrix generate_means () {
        f_matrix u(n_clusters, x_dim);
        // Completar
        return u;
    }

    f_vector generate_prior () {
        vector pi(n_clusters, 1/n_clusters);
        return pi;
    }

    f_matrix calculate_means () {
    
    }

    f_vector calculate_prior () {

    }

    f_matrix calculate_covariance () {

    }

    f_matrix calculate_gamma () {

    }

    f_matrix run (int n_iterations, int aux) {
        f_matrix means, covariance, gamma;
        f_vector prior;

        means = generate_means ();
        prior = generate_prior ();
        covariance = calculate_covariance ();

        gamma = calculate_gamma ();

        for (int it = 0; it < n_iterations, ++it) {

        }

        return gamma;
    }

}