__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.MultiBand.joint_linear import JointLinear
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestJointLinear(object):
    """Tests the source model routines."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "truncation": 5}
        psf_class = PSF(**kwargs_psf)
        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SPEP", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [
            {"ra_source": 0.0001, "dec_source": 0.0, "source_amp": 1.0}
        ]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(
            point_source_type_list=["SOURCE_POSITION"], fixed_magnification_list=[True]
        )
        kwargs_numerics = {"supersampling_factor": 2, "supersampling_convolution": True}
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim
        self.solver = LensEquationSolver(lensModel=lens_model_class)
        multi_band_list = [
            [kwargs_data, kwargs_psf, kwargs_numerics],
            [kwargs_data, kwargs_psf, kwargs_numerics],
        ]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "point_source_model_list": ["SOURCE_POSITION"],
            "fixed_magnification_list": [True],
            "lens_light_model_list": lens_light_model_list,
        }
        self.imageModel = JointLinear(multi_band_list, kwargs_model)

    def test_linear_response(self):
        A = self.imageModel.linear_response_matrix(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
        )
        nx, ny = np.shape(A)
        assert nx == 3
        assert ny == 100**2 * 2

    def test_image_linear_solve(self):
        (
            wls_list,
            model_error_list,
            cov_param,
            param,
        ) = self.imageModel.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=False,
        )
        assert len(wls_list) == 2

    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
        )
        chi2_reduced = logL * 2 / self.imageModel.num_data_evaluate
        npt.assert_almost_equal(chi2_reduced, -1, 1)


if __name__ == "__main__":
    pytest.main()
