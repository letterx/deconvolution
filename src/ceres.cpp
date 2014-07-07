#include "ceres.hpp"

#include "glog/logging.h"
#include <ceres/ceres.h>

class CostFunctionWrapper : public ceres::CostFunction {
 public:
  CostFunctionWrapper(const GradientProblem* problem)
      : problem_(problem) {
    CHECK_NOTNULL(problem_);
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(problem->NumParameters());
  };

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double cost = 0.0;
    bool evaluate_jacobian = (jacobians != nullptr && jacobians[0] != nullptr);
    double* gradient = evaluate_jacobian ? jacobians[0] : nullptr;
    if (!problem_->Evaluate(parameters[0], &cost, gradient)) {
      return false;
    }

    CHECK_GT(cost, 0.0);
    const double r = sqrt(2.0 * cost);
    if (residuals != nullptr) {
      residuals[0] = r;
    }

    if (gradient != nullptr) {
      for (int i = 0; i < parameter_block_sizes()[0]; ++i) {
        gradient[i] /= r;
      }
    }

    return true;
  }

 private:
  const GradientProblem* problem_;
};

ceres::Solver::Summary SolveUsingCeres(const GradientProblem& gradient_problem,
                                       const ceres::Solver::Options& options,
                                       double* parameters) {
  ceres::Problem problem;
  problem.AddResidualBlock(new CostFunctionWrapper(&gradient_problem), NULL, parameters);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return summary;
}


void ceresSolve(GradientProblem& problem, double* parameters) {
    static bool doInit = true;
    if (doInit) {
      google::InitGoogleLogging("ceresSolve");
      doInit = false;
    }

  ceres::Solver::Options options;
  options.max_num_iterations = 500;
  options.minimizer_type = ceres::LINE_SEARCH;
  options.line_search_direction_type = ceres::LBFGS;
  options.line_search_interpolation_type = ceres::CUBIC;
  options.minimizer_progress_to_stdout = true;
  options.max_lbfgs_rank = 20;

  ceres::Solver::Summary summary =
      SolveUsingCeres(problem, options, parameters);

  std::cout << summary.FullReport() << "\n";
}
