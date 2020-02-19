// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCX version 2019.2.0.dev0.
//
// This code was generated with the following parameters:
//
//  {'epsilon': 1e-14,
//   'external_includes': '',
//   'precision': -1,
//   'quadrature_degree': -1,
//   'quadrature_rule': 'auto',
//   'representation': 'auto',
//   'scalar_type': 'double'}


#pragma once

typedef double ufc_scalar_t;
#include <ufc.h>

#ifdef __cplusplus
extern "C" {
#endif

ufc_finite_element* create_ffcx_element_fd9ad35807e1a90cc407af818ab0e05a9bcb08a8_finite_element_main(void);

ufc_finite_element* create_ffcx_element_4af2b623998c933ad5160e4224fac5175247c551_finite_element_main(void);

ufc_dofmap* create_ffcx_element_fd9ad35807e1a90cc407af818ab0e05a9bcb08a8_dofmap_main(void);

ufc_dofmap* create_ffcx_element_4af2b623998c933ad5160e4224fac5175247c551_dofmap_main(void);

ufc_coordinate_mapping* create_ffcx_coordinate_mapping_d91a5290edfe2de70c179383041e8c80ee4d8ddb_coordinate_mapping_main(void);

ufc_integral* create_poisson_cell_integral_ee6168bd65c238da1c979290f4db5f3fd9aa560f_otherwise(void);

ufc_integral* create_poisson_cell_integral_2705ecdaa5354fe831eea765a72683b79258dad9_otherwise(void);

ufc_integral* create_poisson_exterior_facet_integral_2705ecdaa5354fe831eea765a72683b79258dad9_otherwise(void);

ufc_form* create_poisson_form_ee6168bd65c238da1c979290f4db5f3fd9aa560f(void);

ufc_form* create_poisson_form_2705ecdaa5354fe831eea765a72683b79258dad9(void);


// Typedefs for convenience pointers to functions (factories)
typedef ufc_function_space* (*ufc_function_space_factory_ptr)(void);
typedef ufc_form* (*ufc_form_factory_ptr)(void);

// Coefficient spaces helpers (number: 2)
ufc_function_space* poisson_coefficientspace_f_create(void);
ufc_function_space* poisson_coefficientspace_g_create(void);

// Form function spaces helpers (form 'a')
ufc_function_space* poisson_form_a_functionspace_0_create(void);

ufc_function_space* poisson_form_a_functionspace_1_create(void);

/* Coefficient function space typedefs for form "form_a" */
/*   - No form coefficients */
/*    Form helper */
static const ufc_form_factory_ptr poisson_form_a_create = create_poisson_form_ee6168bd65c238da1c979290f4db5f3fd9aa560f;

/*    Typedefs (function spaces for form_a) */
static const ufc_function_space_factory_ptr poisson_form_a_testspace_create = poisson_form_a_functionspace_0_create;
static const ufc_function_space_factory_ptr poisson_form_a_trialspace_create = poisson_form_a_functionspace_1_create;


 /* End coefficient typedefs */

// Form function spaces helpers (form 'L')
ufc_function_space* poisson_form_L_functionspace_0_create(void);

/* Coefficient function space typedefs for form "form_L" */
static const ufc_function_space_factory_ptr poisson_form_L_functionspace_1_create = poisson_coefficientspace_f_create;
static const ufc_function_space_factory_ptr poisson_form_L_functionspace_2_create = poisson_coefficientspace_g_create;
/*    Form helper */
static const ufc_form_factory_ptr poisson_form_L_create = create_poisson_form_2705ecdaa5354fe831eea765a72683b79258dad9;

/*    Typedefs (function spaces for form_L) */
static const ufc_function_space_factory_ptr poisson_form_L_testspace_create = poisson_form_L_functionspace_0_create;
// static ufc_function_space_factory_ptr poisson_form_L_coefficientspace_f_create = poisson_form_L__functionspace_1_create;
// static ufc_function_space_factory_ptr poisson_form_L_coefficientspace_g_create = poisson_form_L__functionspace_2_create;

 /* End coefficient typedefs */

/* Start high-level typedefs */
static const ufc_form_factory_ptr poisson_bilinearform_create = poisson_form_a_create;
static const ufc_form_factory_ptr poisson_linearform_create = poisson_form_L_create;

static const ufc_function_space_factory_ptr poisson_functionspace_create = poisson_form_a_functionspace_0_create;
/* End high-level typedefs */


#ifdef __cplusplus
}
#endif