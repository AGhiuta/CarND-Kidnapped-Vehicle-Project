/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	// Set the number of particles
	num_particles = 100;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; ++i) {
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for(int i = 0; i < num_particles; ++i) {
		double vel_over_yaw = velocity / yaw_rate;
		double yaw_times_t = yaw_rate * delta_t;

		if (fabs(yaw_rate) < 0.00001) {  
      		particles[i].x += velocity * delta_t * cos(particles[i].theta);
      		particles[i].y += velocity * delta_t * sin(particles[i].theta);
    	} else { 
			particles[i].x += vel_over_yaw * (sin(particles[i].theta + yaw_times_t) -
				sin(particles[i].theta));
			particles[i].y += vel_over_yaw * (cos(particles[i].theta) -
				cos(particles[i].theta + yaw_times_t));
			particles[i].theta += yaw_times_t;
		}

		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i = 0; i < observations.size(); ++i) {
		LandmarkObs obs = observations[i];

		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		for(int j = 0; j < predicted.size(); ++j) {
			LandmarkObs pred = predicted[j];
			double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

			if(cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = pred.id;
			}
		}

		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i = 0; i < num_particles; ++i) {
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;

		// Keep only landmarks within sensor range
		vector<LandmarkObs> predictions;

		for(int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int id_i = map_landmarks.landmark_list[j].id_i;
			float x_f = map_landmarks.landmark_list[j].x_f;
			float y_f = map_landmarks.landmark_list[j].y_f;

			if(fabs(x_f - x_p) <= sensor_range && fabs(y_f - y_p) <= sensor_range) {
				predictions.push_back(LandmarkObs{id_i, x_f, y_f});
			}
		}

		// Transform observations
		vector<LandmarkObs> transformed_obs;

		for(int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs = observations[j];

			double t_x = x_p + cos(theta_p) * obs.x - sin(theta_p) * obs.y;
			double t_y = y_p + sin(theta_p) * obs.x + cos(theta_p) * obs.y;

			transformed_obs.push_back(LandmarkObs{obs.id, t_x, t_y});
		}

		dataAssociation(predictions, transformed_obs);

		for(int j = 0; j < transformed_obs.size(); j++) {
			double x_pred, y_pred;
			LandmarkObs obs = transformed_obs[j];

			// Get the x,y coordinates of the prediction associated with the current observation
			for(int k = 0; k < predictions.size(); k++) {
				if(predictions[k].id == obs.id) {
					x_pred = predictions[k].x;
					y_pred = predictions[k].y;
        		}
      		}

      		// Compute weight for this observation with multivariate Gaussian
      		double std_x = std_landmark[0];
      		double std_y = std_landmark[1];
      		double w = 1.0 / (2.0 * M_PI * std_x * std_y) *
      				   exp(-pow(x_pred - obs.x, 2) / (2 * std_x * std_x) -
      				   		pow(y_pred - obs.y, 2) / (2 * std_y * std_y));

			// product of this obersvation weight with total observations weight
			particles[i].weight = w;
    	}
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled;
	default_random_engine gen;

	// get all of the current weights
	for (int i = 0; i < num_particles; ++i) {
		weights[i] = particles[i].weight;
	}

	// generate random starting index for resampling wheel
	int index = rand() % num_particles;

	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> dist_weight(0.0, max_weight);

	double beta = 0.0;

	for(int i = 0; i < num_particles; ++i) {
		beta += dist_weight(gen) * 2.0;

		while(beta > weights[index]) {
	  		beta -= weights[index];
	  		index = (index + 1) % num_particles;
		}
		
		resampled.push_back(particles[index]);
	}

	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
