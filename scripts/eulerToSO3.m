function R = eulerToSO3(phi, theta, psi)
    % Convert Euler angles to SO(3) rotation matrix
    
    
    % Calculate cosine and sine values
    c1 = cos(phi);
    s1 = sin(phi);
    c2 = cos(theta);
    s2 = sin(theta);
    c3 = cos(psi);
    s3 = sin(psi);
    
    % Calculate the rotation matrix
    R = [c1*c2*c3 - s1*s3, -c1*c2*s3 - s1*c3, c1*s2;
         s1*c2*c3 + c1*s3, -s1*c2*s3 + c1*c3, s1*s2;
         -s2*c3, s2*s3, c2];
end