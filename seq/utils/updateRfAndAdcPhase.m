function [rf, adc, rf_phase, rf_inc] = updateRfAndAdcPhase(rf, adc, rf_phase, rf_inc, rfSpoilingInc)
    rf.phaseOffset = rf_phase / 180 * pi;
    adc.phaseOffset = rf.phaseOffset;
    rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
    rf_phase = mod(rf_phase + rf_inc, 360.0);
end
