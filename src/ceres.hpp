#ifndef _CERES_HPP_
#define _CERES_HPP_

class GradientProblem {
 public:
  virtual ~GradientProblem() {}
  virtual int NumParameters() const = 0;
  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const = 0;
};

void ceresSolve(GradientProblem& problem, double* parameters);

#endif
