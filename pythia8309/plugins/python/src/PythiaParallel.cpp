#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/Event.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/LesHouches.h>
#include <Pythia8/Merging.h>
#include <Pythia8/MergingHooks.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ParticleDecays.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/PartonVertex.h>
#include <Pythia8/PhaseSpace.h>
#include <Pythia8/PhysicsBase.h>
#include <Pythia8/Pythia.h>
#include <Pythia8/PythiaParallel.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SharedPointers.h>
#include <Pythia8/ShowerModel.h>
#include <Pythia8/SigmaLowEnergy.h>
#include <Pythia8/SigmaProcess.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/UserHooks.h>
#include <Pythia8/Weights.h>
#include <functional>
#include <ios>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream> // __str__
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <Pythia8/UserHooks.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/BeamShape.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// Pythia8::HeavyIons file:Pythia8/HeavyIons.h line:31
struct PyCallBack_Pythia8_HeavyIons : public Pythia8::HeavyIons {
	using Pythia8::HeavyIons::HeavyIons;

	bool init() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "init");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"HeavyIons::init\"");
	}
	bool next() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "next");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"HeavyIons::next\"");
	}
	void stat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "stat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return HeavyIons::stat();
	}
	void onInitInfoPtr() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "onInitInfoPtr");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onInitInfoPtr();
	}
	void onBeginEvent() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "onBeginEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onBeginEvent();
	}
	void onEndEvent(enum Pythia8::PhysicsBase::Status a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "onEndEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onEndEvent(a0);
	}
	void onStat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons *>(this), "onStat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onStat();
	}
};

// Pythia8::HeavyIons::InfoGrabber file:Pythia8/HeavyIons.h line:123
struct PyCallBack_Pythia8_HeavyIons_InfoGrabber : public Pythia8::HeavyIons::InfoGrabber {
	using Pythia8::HeavyIons::InfoGrabber::InfoGrabber;

	bool initAfterBeams() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "initAfterBeams");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::initAfterBeams();
	}
	bool canModifySigma() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canModifySigma");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canModifySigma();
	}
	bool canBiasSelection() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canBiasSelection");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canBiasSelection();
	}
	double biasedSelectionWeight() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "biasedSelectionWeight");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::override_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return UserHooks::biasedSelectionWeight();
	}
	bool canVetoProcessLevel() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoProcessLevel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoProcessLevel();
	}
	bool doVetoProcessLevel(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoProcessLevel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoProcessLevel(a0);
	}
	bool canVetoResonanceDecays() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoResonanceDecays");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoResonanceDecays();
	}
	bool doVetoResonanceDecays(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoResonanceDecays");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoResonanceDecays(a0);
	}
	bool canVetoPT() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoPT");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoPT();
	}
	double scaleVetoPT() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "scaleVetoPT");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::override_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return UserHooks::scaleVetoPT();
	}
	bool doVetoPT(int a0, const class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoPT");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoPT(a0, a1);
	}
	bool canVetoStep() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoStep();
	}
	int numberVetoStep() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "numberVetoStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return UserHooks::numberVetoStep();
	}
	bool doVetoStep(int a0, int a1, int a2, const class Pythia8::Event & a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoStep(a0, a1, a2, a3);
	}
	bool canVetoMPIStep() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoMPIStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoMPIStep();
	}
	int numberVetoMPIStep() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "numberVetoMPIStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return UserHooks::numberVetoMPIStep();
	}
	bool doVetoMPIStep(int a0, const class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoMPIStep");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoMPIStep(a0, a1);
	}
	bool canVetoPartonLevelEarly() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoPartonLevelEarly");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoPartonLevelEarly();
	}
	bool doVetoPartonLevelEarly(const class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoPartonLevelEarly");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoPartonLevelEarly(a0);
	}
	bool retryPartonLevel() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "retryPartonLevel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::retryPartonLevel();
	}
	bool canVetoPartonLevel() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoPartonLevel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoPartonLevel();
	}
	bool doVetoPartonLevel(const class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoPartonLevel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoPartonLevel(a0);
	}
	bool canSetResonanceScale() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canSetResonanceScale");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canSetResonanceScale();
	}
	double scaleResonance(int a0, const class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "scaleResonance");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::override_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return UserHooks::scaleResonance(a0, a1);
	}
	bool canVetoISREmission() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoISREmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoISREmission();
	}
	bool doVetoISREmission(int a0, const class Pythia8::Event & a1, int a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoISREmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoISREmission(a0, a1, a2);
	}
	bool canVetoFSREmission() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoFSREmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoFSREmission();
	}
	bool doVetoFSREmission(int a0, const class Pythia8::Event & a1, int a2, bool a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoFSREmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoFSREmission(a0, a1, a2, a3);
	}
	bool canVetoMPIEmission() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoMPIEmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoMPIEmission();
	}
	bool doVetoMPIEmission(int a0, const class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoMPIEmission");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoMPIEmission(a0, a1);
	}
	bool canReconnectResonanceSystems() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canReconnectResonanceSystems");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canReconnectResonanceSystems();
	}
	bool doReconnectResonanceSystems(int a0, class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doReconnectResonanceSystems");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doReconnectResonanceSystems(a0, a1);
	}
	bool canChangeFragPar() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canChangeFragPar");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canChangeFragPar();
	}
	bool canVetoAfterHadronization() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canVetoAfterHadronization");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canVetoAfterHadronization();
	}
	bool doVetoAfterHadronization(const class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doVetoAfterHadronization");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::doVetoAfterHadronization(a0);
	}
	bool canSetImpactParameter() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "canSetImpactParameter");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return UserHooks::canSetImpactParameter();
	}
	double doSetImpactParameter() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "doSetImpactParameter");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::override_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return UserHooks::doSetImpactParameter();
	}
	void onInitInfoPtr() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "onInitInfoPtr");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return UserHooks::onInitInfoPtr();
	}
	void onBeginEvent() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "onBeginEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onBeginEvent();
	}
	void onEndEvent(enum Pythia8::PhysicsBase::Status a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "onEndEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onEndEvent(a0);
	}
	void onStat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::HeavyIons::InfoGrabber *>(this), "onStat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onStat();
	}
};

void bind_Pythia8_PythiaParallel(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::PythiaParallel file:Pythia8/PythiaParallel.h line:18
		pybind11::class_<Pythia8::PythiaParallel, std::shared_ptr<Pythia8::PythiaParallel>> cl(M("Pythia8"), "PythiaParallel", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::PythiaParallel(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::PythiaParallel(a0); } ), "doc" , pybind11::arg("xmlDir"));
		cl.def( pybind11::init<std::string, bool>(), pybind11::arg("xmlDir"), pybind11::arg("printBanner") );

		cl.def("readString", [](Pythia8::PythiaParallel &o, class std::basic_string<char> const & a0) -> bool { return o.readString(a0); }, "", pybind11::arg("setting"));
		cl.def("readString", (bool (Pythia8::PythiaParallel::*)(std::string, bool)) &Pythia8::PythiaParallel::readString, "C++: Pythia8::PythiaParallel::readString(std::string, bool) --> bool", pybind11::arg("setting"), pybind11::arg("warn"));
		cl.def("readFile", [](Pythia8::PythiaParallel &o, class std::basic_string<char> const & a0) -> bool { return o.readFile(a0); }, "", pybind11::arg("fileName"));
		cl.def("readFile", [](Pythia8::PythiaParallel &o, class std::basic_string<char> const & a0, bool const & a1) -> bool { return o.readFile(a0, a1); }, "", pybind11::arg("fileName"), pybind11::arg("warn"));
		cl.def("readFile", (bool (Pythia8::PythiaParallel::*)(std::string, bool, int)) &Pythia8::PythiaParallel::readFile, "C++: Pythia8::PythiaParallel::readFile(std::string, bool, int) --> bool", pybind11::arg("fileName"), pybind11::arg("warn"), pybind11::arg("subrun"));
		cl.def("readFile", (bool (Pythia8::PythiaParallel::*)(std::string, int)) &Pythia8::PythiaParallel::readFile, "C++: Pythia8::PythiaParallel::readFile(std::string, int) --> bool", pybind11::arg("fileName"), pybind11::arg("subrun"));
		cl.def("readFile", [](Pythia8::PythiaParallel &o) -> bool { return o.readFile(); }, "");
		cl.def("readFile", [](Pythia8::PythiaParallel &o, class std::basic_istream<char> & a0) -> bool { return o.readFile(a0); }, "", pybind11::arg("is"));
		cl.def("readFile", [](Pythia8::PythiaParallel &o, class std::basic_istream<char> & a0, bool const & a1) -> bool { return o.readFile(a0, a1); }, "", pybind11::arg("is"), pybind11::arg("warn"));
		cl.def("readFile", (bool (Pythia8::PythiaParallel::*)(class std::basic_istream<char> &, bool, int)) &Pythia8::PythiaParallel::readFile, "C++: Pythia8::PythiaParallel::readFile(class std::basic_istream<char> &, bool, int) --> bool", pybind11::arg("is"), pybind11::arg("warn"), pybind11::arg("subrun"));
		cl.def("readFile", (bool (Pythia8::PythiaParallel::*)(class std::basic_istream<char> &, int)) &Pythia8::PythiaParallel::readFile, "C++: Pythia8::PythiaParallel::readFile(class std::basic_istream<char> &, int) --> bool", pybind11::arg("is"), pybind11::arg("subrun"));
		cl.def("init", (bool (Pythia8::PythiaParallel::*)()) &Pythia8::PythiaParallel::init, "C++: Pythia8::PythiaParallel::init() --> bool");
		cl.def("init", (bool (Pythia8::PythiaParallel::*)(class std::function<bool (class Pythia8::Pythia *)>)) &Pythia8::PythiaParallel::init, "C++: Pythia8::PythiaParallel::init(class std::function<bool (class Pythia8::Pythia *)>) --> bool", pybind11::arg("additionalSetup"));
		cl.def("foreach", (void (Pythia8::PythiaParallel::*)(class std::function<void (class Pythia8::Pythia *)>)) &Pythia8::PythiaParallel::foreach, "C++: Pythia8::PythiaParallel::foreach(class std::function<void (class Pythia8::Pythia *)>) --> void", pybind11::arg("action"));
		cl.def("foreachAsync", (void (Pythia8::PythiaParallel::*)(class std::function<void (class Pythia8::Pythia *)>)) &Pythia8::PythiaParallel::foreachAsync, "C++: Pythia8::PythiaParallel::foreachAsync(class std::function<void (class Pythia8::Pythia *)>) --> void", pybind11::arg("action"));
		cl.def("stat", (void (Pythia8::PythiaParallel::*)()) &Pythia8::PythiaParallel::stat, "C++: Pythia8::PythiaParallel::stat() --> void");
		cl.def("run", (class std::vector<long, class std::allocator<long> > (Pythia8::PythiaParallel::*)(long, class std::function<void (class Pythia8::Pythia *)>)) &Pythia8::PythiaParallel::run, pybind11::call_guard<pybind11::gil_scoped_release>(), "C++: Pythia8::PythiaParallel::run(long, class std::function<void (class Pythia8::Pythia *)>) --> class std::vector<long, class std::allocator<long> >", pybind11::arg("nEvents"), pybind11::arg("callback"));
		cl.def("run", (class std::vector<long, class std::allocator<long> > (Pythia8::PythiaParallel::*)(class std::function<void (class Pythia8::Pythia *)>)) &Pythia8::PythiaParallel::run, pybind11::call_guard<pybind11::gil_scoped_release>(), "C++: Pythia8::PythiaParallel::run(class std::function<void (class Pythia8::Pythia *)>) --> class std::vector<long, class std::allocator<long> >", pybind11::arg("callback"));
		cl.def("sigmaGen", (double (Pythia8::PythiaParallel::*)() const) &Pythia8::PythiaParallel::sigmaGen, "C++: Pythia8::PythiaParallel::sigmaGen() const --> double");
		cl.def("weightSum", (double (Pythia8::PythiaParallel::*)() const) &Pythia8::PythiaParallel::weightSum, "C++: Pythia8::PythiaParallel::weightSum() const --> double");
	}
	{ // Pythia8::HeavyIons file:Pythia8/HeavyIons.h line:31
		pybind11::class_<Pythia8::HeavyIons, std::shared_ptr<Pythia8::HeavyIons>, PyCallBack_Pythia8_HeavyIons> cl(M("Pythia8"), "HeavyIons", "");
		pybind11::handle cl_type = cl;

		{ // Pythia8::HeavyIons::InfoGrabber file:Pythia8/HeavyIons.h line:123
			auto & enclosing_class = cl;
			pybind11::class_<Pythia8::HeavyIons::InfoGrabber, std::shared_ptr<Pythia8::HeavyIons::InfoGrabber>, PyCallBack_Pythia8_HeavyIons_InfoGrabber, Pythia8::UserHooks> cl(enclosing_class, "InfoGrabber", "");
			pybind11::handle cl_type = cl;

			cl.def( pybind11::init( [](){ return new Pythia8::HeavyIons::InfoGrabber(); }, [](){ return new PyCallBack_Pythia8_HeavyIons_InfoGrabber(); } ) );
			cl.def("getInfo", (class Pythia8::Info * (Pythia8::HeavyIons::InfoGrabber::*)()) &Pythia8::HeavyIons::InfoGrabber::getInfo, "C++: Pythia8::HeavyIons::InfoGrabber::getInfo() --> class Pythia8::Info *", pybind11::return_value_policy::automatic);
			cl.def("assign", (struct Pythia8::HeavyIons::InfoGrabber & (Pythia8::HeavyIons::InfoGrabber::*)(const struct Pythia8::HeavyIons::InfoGrabber &)) &Pythia8::HeavyIons::InfoGrabber::operator=, "C++: Pythia8::HeavyIons::InfoGrabber::operator=(const struct Pythia8::HeavyIons::InfoGrabber &) --> struct Pythia8::HeavyIons::InfoGrabber &", pybind11::return_value_policy::reference, pybind11::arg(""));
		}

		cl.def( pybind11::init<class Pythia8::Pythia &>(), pybind11::arg("mainPythiaIn") );

		cl.def_readwrite("sigtot", &Pythia8::HeavyIons::sigtot);
		cl.def_readwrite("HIHooksPtr", &Pythia8::HeavyIons::HIHooksPtr);
		cl.def_readwrite("pythia", &Pythia8::HeavyIons::pythia);
		cl.def_readwrite("pythiaNames", &Pythia8::HeavyIons::pythiaNames);
		cl.def_readwrite("info", &Pythia8::HeavyIons::info);
		cl.def_readwrite("hiInfo", &Pythia8::HeavyIons::hiInfo);
		cl.def("init", (bool (Pythia8::HeavyIons::*)()) &Pythia8::HeavyIons::init, "C++: Pythia8::HeavyIons::init() --> bool");
		cl.def("next", (bool (Pythia8::HeavyIons::*)()) &Pythia8::HeavyIons::next, "C++: Pythia8::HeavyIons::next() --> bool");
		cl.def_static("addSpecialSettings", (void (*)(class Pythia8::Settings &)) &Pythia8::HeavyIons::addSpecialSettings, "C++: Pythia8::HeavyIons::addSpecialSettings(class Pythia8::Settings &) --> void", pybind11::arg("settings"));
		cl.def_static("isHeavyIon", (bool (*)(class Pythia8::Settings &)) &Pythia8::HeavyIons::isHeavyIon, "C++: Pythia8::HeavyIons::isHeavyIon(class Pythia8::Settings &) --> bool", pybind11::arg("settings"));
		cl.def("setHIUserHooksPtr", (bool (Pythia8::HeavyIons::*)(class std::shared_ptr<class Pythia8::HIUserHooks>)) &Pythia8::HeavyIons::setHIUserHooksPtr, "C++: Pythia8::HeavyIons::setHIUserHooksPtr(class std::shared_ptr<class Pythia8::HIUserHooks>) --> bool", pybind11::arg("userHooksPtrIn"));
		cl.def("sumUpMessages", (void (Pythia8::HeavyIons::*)(class Pythia8::Info &, std::string, const class Pythia8::Info *)) &Pythia8::HeavyIons::sumUpMessages, "C++: Pythia8::HeavyIons::sumUpMessages(class Pythia8::Info &, std::string, const class Pythia8::Info *) --> void", pybind11::arg("in"), pybind11::arg("tag"), pybind11::arg("other"));
		cl.def("updateInfo", (void (Pythia8::HeavyIons::*)()) &Pythia8::HeavyIons::updateInfo, "C++: Pythia8::HeavyIons::updateInfo() --> void");
		cl.def("clearProcessLevel", (void (Pythia8::HeavyIons::*)(class Pythia8::Pythia &)) &Pythia8::HeavyIons::clearProcessLevel, "C++: Pythia8::HeavyIons::clearProcessLevel(class Pythia8::Pythia &) --> void", pybind11::arg("pyt"));
		cl.def_static("setupSpecials", (void (*)(class Pythia8::Settings &, std::string)) &Pythia8::HeavyIons::setupSpecials, "C++: Pythia8::HeavyIons::setupSpecials(class Pythia8::Settings &, std::string) --> void", pybind11::arg("settings"), pybind11::arg("match"));
		cl.def_static("setupSpecials", (void (*)(class Pythia8::Pythia &, std::string)) &Pythia8::HeavyIons::setupSpecials, "C++: Pythia8::HeavyIons::setupSpecials(class Pythia8::Pythia &, std::string) --> void", pybind11::arg("p"), pybind11::arg("match"));
		cl.def("stat", (void (Pythia8::HeavyIons::*)()) &Pythia8::HeavyIons::stat, "C++: Pythia8::HeavyIons::stat() --> void");
		cl.def("assign", (class Pythia8::HeavyIons & (Pythia8::HeavyIons::*)(const class Pythia8::HeavyIons &)) &Pythia8::HeavyIons::operator=, "C++: Pythia8::HeavyIons::operator=(const class Pythia8::HeavyIons &) --> class Pythia8::HeavyIons &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
}
