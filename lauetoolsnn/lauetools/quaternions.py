# -*- coding: utf-8 -*-
"""
# Copyright 2011-16 Max-Planck-Institut für Eisenforschung GmbH
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

Created on Thu Nov 10 23:39:06 2022

@author: PURUSHOT

Quaternion class for LaueNN module

This file is in its entirety taken from DAMASK package
https://github.com/puwe/DAMASK

small modifications were done to convert from python2.x to 3.x
"""

import math
import numpy as np

# ******************************************************************************************
class Rodrigues:
    def __init__(self, vector = np.zeros(3)):
      self.vector = vector
    def asQuaternion(self):
      norm = np.linalg.norm(self.vector)
      halfAngle = np.arctan(norm)
      return Quaternion(np.cos(halfAngle),np.sin(halfAngle)*self.vector/norm)
    def asAngleAxis(self):
      norm = np.linalg.norm(self.vector)
      halfAngle = np.arctan(norm)
      return (2.0*halfAngle,self.vector/norm)

def cmp(a, b):
    return (a > b) - (a < b) 
# ******************************************************************************************
class Quaternion:
    """
    Orientation represented as unit quaternion
    All methods and naming conventions based on http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions
    w is the real part, (x, y, z) are the imaginary parts
    Representation of rotation is in ACTIVE form!
    (derived directly or through angleAxis, Euler angles, or active matrix)
    vector "a" (defined in coordinate system "A") is actively rotated to new coordinates "b"
    b = Q * a
    b = np.dot(Q.asMatrix(),a)
    """

    def __init__(self, quatArray = [1.0,0.0,0.0,0.0]):
        """initializes to identity if not given"""
        self.w, \
        self.x, \
        self.y, \
        self.z = quatArray
        self.homomorph()

    def __iter__(self):
        """components"""
        return iter([self.w,self.x,self.y,self.z])

    def __copy__(self):
        """create copy"""
        Q = Quaternion([self.w,self.x,self.y,self.z])
        return Q

    copy = __copy__

    def __repr__(self):
        """readbable string"""
        return 'Quaternion(real=%+.6f, imag=<%+.6f, %+.6f, %+.6f>)' % \
            (self.w, self.x, self.y, self.z)

    def __pow__(self, exponent):
        """power"""
        omega = math.acos(self.w)
        vRescale = math.sin(exponent*omega)/math.sin(omega)
        Q = Quaternion()
        Q.w = math.cos(exponent*omega)
        Q.x = self.x * vRescale
        Q.y = self.y * vRescale
        Q.z = self.z * vRescale
        return Q

    def __ipow__(self, exponent):
        """in place power"""
        omega    = math.acos(self.w)
        vRescale = math.sin(exponent*omega)/math.sin(omega)
        self.w  = np.cos(exponent*omega)
        self.x *= vRescale
        self.y *= vRescale
        self.z *= vRescale
        return self

    def __mul__(self, other):
        """multiplication"""
        try:         # quaternion
            Aw = self.w
            Ax = self.x
            Ay = self.y
            Az = self.z
            Bw = other.w
            Bx = other.x
            By = other.y
            Bz = other.z
            Q = Quaternion()
            Q.w = - Ax * Bx - Ay * By - Az * Bz + Aw * Bw
            Q.x = + Ax * Bw + Ay * Bz - Az * By + Aw * Bx
            Q.y = - Ax * Bz + Ay * Bw + Az * Bx + Aw * By
            Q.z = + Ax * By - Ay * Bx + Az * Bw + Aw * Bz
            return Q
        except: 
            pass
        try:                                                         # vector (perform active rotation, i.e. q*v*q.conjugated)
            w = self.w
            x = self.x
            y = self.y
            z = self.z
            Vx = other[0]
            Vy = other[1]
            Vz = other[2]
    
            return np.array([\
               w * w * Vx + 2 * y * w * Vz - 2 * z * w * Vy + \
               x * x * Vx + 2 * y * x * Vy + 2 * z * x * Vz - \
               z * z * Vx - y * y * Vx,
               2 * x * y * Vx + y * y * Vy + 2 * z * y * Vz + \
               2 * w * z * Vx - z * z * Vy + w * w * Vy - \
               2 * x * w * Vz - x * x * Vy,
               2 * x * z * Vx + 2 * y * z * Vy + \
               z * z * Vz - 2 * w * y * Vx - y * y * Vz + \
               2 * w * x * Vy - x * x * Vz + w * w * Vz ])
        except: 
            pass
        try:                                                        # scalar
            Q = self.copy()
            Q.w *= other
            Q.x *= other
            Q.y *= other
            Q.z *= other
            return Q
        except:
            return self.copy()

    def __imul__(self, other):
      """in place multiplication"""
      try:                                                        # Quaternion
          Ax = self.x
          Ay = self.y
          Az = self.z
          Aw = self.w
          Bx = other.x
          By = other.y
          Bz = other.z
          Bw = other.w
          self.x =  Ax * Bw + Ay * Bz - Az * By + Aw * Bx
          self.y = -Ax * Bz + Ay * Bw + Az * Bx + Aw * By
          self.z =  Ax * By - Ay * Bx + Az * Bw + Aw * Bz
          self.w = -Ax * Bx - Ay * By - Az * Bz + Aw * Bw
      except: 
          pass
      return self

    def __div__(self, other):
        """division"""
        if isinstance(other, (int,float)):
            w = self.w / other
            x = self.x / other
            y = self.y / other
            z = self.z / other
            return self.__class__([w,x,y,z])
        else:
            return NotImplemented

    def __idiv__(self, other):
        """in place division"""
        if isinstance(other, (int,float)):
            self.w /= other
            self.x /= other
            self.y /= other
            self.z /= other
        return self

    def __add__(self, other):
        """addition"""
        if isinstance(other, Quaternion):
            w = self.w + other.w
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
            return self.__class__([w,x,y,z])
        else:
            return NotImplemented

    def __iadd__(self, other):
        """in place division"""
        if isinstance(other, Quaternion):
            self.w += other.w
            self.x += other.x
            self.y += other.y
            self.z += other.z
        return self

    def __sub__(self, other):
        """subtraction"""
        if isinstance(other, Quaternion):
            Q = self.copy()
            Q.w -= other.w
            Q.x -= other.x
            Q.y -= other.y
            Q.z -= other.z
            return Q
        else:
            return self.copy()

    def __isub__(self, other):
        """in place subtraction"""
        if isinstance(other, Quaternion):
            self.w -= other.w
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        return self

    def __neg__(self):
        """additive inverse"""
        self.w = -self.w
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def __abs__(self):
        """norm"""
        return math.sqrt(self.w ** 2 + \
                         self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    magnitude = __abs__

    def __eq__(self,other):
        """equal at e-8 precision"""
        return (abs(self.w-other.w) < 1e-8 and \
                abs(self.x-other.x) < 1e-8 and \
                abs(self.y-other.y) < 1e-8 and \
                abs(self.z-other.z) < 1e-8) \
                or \
               (abs(-self.w-other.w) < 1e-8 and \
                abs(-self.x-other.x) < 1e-8 and \
                abs(-self.y-other.y) < 1e-8 and \
                abs(-self.z-other.z) < 1e-8)

    def __ne__(self,other):
        """not equal at e-8 precision"""
        return not self.__eq__(self,other)

    def __cmp__(self,other):
        """linear ordering"""
        return cmp(self.Rodrigues(),other.Rodrigues())

    def magnitude_squared(self):
        return self.w ** 2 + \
               self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2

    def identity(self):
        self.w = 1.
        self.x = 0.
        self.y = 0.
        self.z = 0.
        return self

    def normalize(self):
        d = self.magnitude()
        if d > 0.0:
            self.w /= d
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def conjugate(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def inverse(self):
        d = self.magnitude()
        if d > 0.0:
          self.conjugate()
          self.w /= d
          self.x /= d
          self.y /= d
          self.z /= d
        return self

    def homomorph(self):
        if self.w < 0.0:
          self.w = -self.w
          self.x = -self.x
          self.y = -self.y
          self.z = -self.z
        return self

    def normalized(self):
        return self.copy().normalize()

    def conjugated(self):
        return self.copy().conjugate()

    def inversed(self):
        return self.copy().inverse()

    def homomorphed(self):
        return self.copy().homomorph()

    def asList(self):
        return [i for i in self]

    def asM(self):                                                # to find Averaging Quaternions (see F. Landis Markley et al.)
        return np.outer([i for i in self],[i for i in self])

    def asMatrix(self):
        return np.array(
          [[1.0-2.0*(self.y*self.y+self.z*self.z),     2.0*(self.x*self.y-self.z*self.w),     2.0*(self.x*self.z+self.y*self.w)],
           [    2.0*(self.x*self.y+self.z*self.w), 1.0-2.0*(self.x*self.x+self.z*self.z),     2.0*(self.y*self.z-self.x*self.w)],
           [    2.0*(self.x*self.z-self.y*self.w),     2.0*(self.x*self.w+self.y*self.z), 1.0-2.0*(self.x*self.x+self.y*self.y)]])

    def asAngleAxis(self, degrees = False):
        if self.w > 1:
            self.normalize()
    
        s = math.sqrt(1. - self.w**2)
        x = 2*self.w**2 - 1.
        y = 2*self.w * s
    
        angle = math.atan2(y,x)
        if angle < 0.0:
            angle *= -1.
            s     *= -1.
    
        return (np.degrees(angle) if degrees else angle,
                np.array([1.0, 0.0, 0.0] if np.abs(angle) < 1e-6 else [self.x / s, self.y / s, self.z / s]))

    def asRodrigues(self):
        return np.inf*np.ones(3) if self.w == 0.0 else np.array([self.x, self.y, self.z])/self.w

    def asEulers(self,
                 type = "bunge",
                 degrees = False,
                 standardRange = False):
        """
        Orientation as Bunge-Euler angles
    
        conversion of ACTIVE rotation to Euler angles taken from:
        Melcher, A.; Unser, A.; Reichhardt, M.; Nestler, B.; Poetschke, M.; Selzer, M.
        Conversion of EBSD data by a quaternion based algorithm to be used for grain structure simulations
        Technische Mechanik 30 (2010) pp 401--413
        """
        angles = [0.0,0.0,0.0]
    
        if type.lower() == 'bunge' or type.lower() == 'zxz':
            if   abs(self.x) < 1e-4 and abs(self.y) < 1e-4:
                x = self.w**2 - self.z**2
                y = 2.*self.w*self.z
                angles[0] = math.atan2(y,x)
            elif abs(self.w) < 1e-4 and abs(self.z) < 1e-4:
                x = self.x**2 - self.y**2
                y = 2.*self.x*self.y
                angles[0] = math.atan2(y,x)
                angles[1] = math.pi
            else:
                chi = math.sqrt((self.w**2 + self.z**2)*(self.x**2 + self.y**2))
        
                x = (self.w * self.x - self.y * self.z)/2./chi
                y = (self.w * self.y + self.x * self.z)/2./chi
                angles[0] = math.atan2(y,x)
        
                x = self.w**2 + self.z**2 - (self.x**2 + self.y**2)
                y = 2.*chi
                angles[1] = math.atan2(y,x)
        
                x = (self.w * self.x + self.y * self.z)/2./chi
                y = (self.z * self.x - self.y * self.w)/2./chi
                angles[2] = math.atan2(y,x)
      
            if standardRange:
                angles[0] %= 2*math.pi
                if angles[1] < 0.0:
                  angles[1] += math.pi
                  angles[2] *= -1.0
                angles[2] %= 2*math.pi
        return np.degrees(angles) if degrees else angles

#    # Static constructors
    @classmethod
    def fromIdentity(cls):
        return cls()


    @classmethod
    def fromRandom(cls,randomSeed = None):
        if randomSeed is None:
            randomSeed = np.random.randint(1e6)
        np.random.seed(randomSeed)
        r = np.random.random(3)
        w = math.cos(2.0*math.pi*r[0])*math.sqrt(r[2])
        x = math.sin(2.0*math.pi*r[1])*math.sqrt(1.0-r[2])
        y = math.cos(2.0*math.pi*r[1])*math.sqrt(1.0-r[2])
        z = math.sin(2.0*math.pi*r[0])*math.sqrt(r[2])
        return cls([w,x,y,z])


    @classmethod
    def fromRodrigues(cls, rodrigues):
        if not isinstance(rodrigues, np.ndarray): rodrigues = np.array(rodrigues)
        halfangle = math.atan(np.linalg.norm(rodrigues))
        c = math.cos(halfangle)
        w = c
        x,y,z = c*rodrigues
        return cls([w,x,y,z])


    @classmethod
    def fromAngleAxis(cls, angle, axis):
        if not isinstance(axis, np.ndarray): axis = np.array(axis,dtype='d')
        axis = axis.astype(float)/np.linalg.norm(axis)
        s = math.sin(angle / 2.0)
        w = math.cos(angle / 2.0)
        x = axis[0] * s
        y = axis[1] * s
        z = axis[2] * s
        return cls([w,x,y,z])


    @classmethod
    def fromEulers(cls, eulers, type = 'Bunge'):

        eulers *= 0.5                                                                  # reduce to half angles
    
        c1 = math.cos(eulers[0])
        s1 = math.sin(eulers[0])
        c2 = math.cos(eulers[1])
        s2 = math.sin(eulers[1])
        c3 = math.cos(eulers[2])
        s3 = math.sin(eulers[2])
    
        if type.lower() == 'bunge' or type.lower() == 'zxz':
            w =   c1 * c2 * c3 - s1 * c2 * s3
            x =   c1 * s2 * c3 + s1 * s2 * s3
            y = - c1 * s2 * s3 + s1 * s2 * c3
            z =   c1 * c2 * s3 + s1 * c2 * c3
        else:
            w = c1 * c2 * c3 - s1 * s2 * s3
            x = s1 * s2 * c3 + c1 * c2 * s3
            y = s1 * c2 * c3 + c1 * s2 * s3
            z = c1 * s2 * c3 - s1 * c2 * s3
        return cls([w,x,y,z])


# Modified Method to calculate Quaternion from Orientation Matrix,
# Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    @classmethod
    def fromMatrix(cls, m):
        if m.shape != (3,3) and np.prod(m.shape) == 9:
          m = m.reshape(3,3)
    
        tr = np.trace(m)
        if tr > 1e-8:
          s = math.sqrt(tr + 1.0)*2.0
    
          return cls(
            [ s*0.25,
              (m[2,1] - m[1,2])/s,
              (m[0,2] - m[2,0])/s,
              (m[1,0] - m[0,1])/s,
            ])
    
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
          t = m[0,0] - m[1,1] - m[2,2] + 1.0
          s = 2.0*math.sqrt(t)
    
          return cls(
            [ (m[2,1] - m[1,2])/s,
              s*0.25,
              (m[0,1] + m[1,0])/s,
              (m[2,0] + m[0,2])/s,
            ])
    
        elif m[1,1] > m[2,2]:
          t = -m[0,0] + m[1,1] - m[2,2] + 1.0
          s = 2.0*math.sqrt(t)
    
          return cls(
            [ (m[0,2] - m[2,0])/s,
              (m[0,1] + m[1,0])/s,
              s*0.25,
              (m[1,2] + m[2,1])/s,
            ])
    
        else:
          t = -m[0,0] - m[1,1] + m[2,2] + 1.0
          s = 2.0*math.sqrt(t)
    
          return cls(
            [ (m[1,0] - m[0,1])/s,
              (m[2,0] + m[0,2])/s,
              (m[1,2] + m[2,1])/s,
              s*0.25,
            ])


    @classmethod
    def new_interpolate(cls, q1, q2, t):
        """
        interpolation
 
        see http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872_2007014421.pdf
        for (another?) way to interpolate quaternions
        """
        assert isinstance(q1, Quaternion) and isinstance(q2, Quaternion)
        Q = cls()

        costheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        if costheta < 0.:
            costheta = -costheta
            q1 = q1.conjugated()
        elif costheta > 1:
            costheta = 1

        theta = math.acos(costheta)
        if abs(theta) < 0.01:
            Q.w = q2.w
            Q.x = q2.x
            Q.y = q2.y
            Q.z = q2.z
            return Q

        sintheta = math.sqrt(1.0 - costheta * costheta)
        if abs(sintheta) < 0.01:
            Q.w = (q1.w + q2.w) * 0.5
            Q.x = (q1.x + q2.x) * 0.5
            Q.y = (q1.y + q2.y) * 0.5
            Q.z = (q1.z + q2.z) * 0.5
            return Q

        ratio1 = math.sin((1 - t) * theta) / sintheta
        ratio2 = math.sin(t * theta) / sintheta

        Q.w = q1.w * ratio1 + q2.w * ratio2
        Q.x = q1.x * ratio1 + q2.x * ratio2
        Q.y = q1.y * ratio1 + q2.y * ratio2
        Q.z = q1.z * ratio1 + q2.z * ratio2
        return Q


# ******************************************************************************************
class Symmetry:

  lattices = [None,'orthorhombic','tetragonal','hexagonal','cubic',]

  def __init__(self, symmetry = None):
    """lattice with given symmetry, defaults to None"""
    if isinstance(symmetry, str) and symmetry.lower() in Symmetry.lattices:
      self.lattice = symmetry.lower()
    else:
      self.lattice = None


  def __copy__(self):
    """copy"""
    return self.__class__(self.lattice)

  copy = __copy__


  def __repr__(self):
    """readbable string"""
    return '%s' % (self.lattice)


  def __eq__(self, other):
    """equal"""
    return self.lattice == other.lattice

  def __neq__(self, other):
    """not equal"""
    return not self.__eq__(other)

  def __cmp__(self,other):
    """linear ordering"""
    return cmp(Symmetry.lattices.index(self.lattice),Symmetry.lattices.index(other.lattice))

  def symmetryQuats(self,who = []):
    """List of symmetry operations as quaternions."""
    if self.lattice == 'cubic':
      symQuats =  [
                    [ 1.0,              0.0,              0.0,              0.0              ],
                    [ 0.0,              1.0,              0.0,              0.0              ],
                    [ 0.0,              0.0,              1.0,              0.0              ],
                    [ 0.0,              0.0,              0.0,              1.0              ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2), 0.5*math.sqrt(2) ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.0,             -0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,             -0.5              ],
                    [-0.5,              0.5,             -0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,             -0.5              ],
                    [-0.5,             -0.5,             -0.5,              0.5              ],
                    [-0.5,              0.5,             -0.5,             -0.5              ],
                    [-0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [ 0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [-0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0,              0.0              ],
                    [-0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0,              0.0              ],
                  ]
    elif self.lattice == 'hexagonal':
      symQuats =  [
                    [ 1.0,0.0,0.0,0.0 ],
                    [-0.5*math.sqrt(3), 0.0, 0.0,-0.5 ],
                    [ 0.5, 0.0, 0.0, 0.5*math.sqrt(3) ],
                    [ 0.0,0.0,0.0,1.0 ],
                    [-0.5, 0.0, 0.0, 0.5*math.sqrt(3) ],
                    [-0.5*math.sqrt(3), 0.0, 0.0, 0.5 ],
                    [ 0.0,1.0,0.0,0.0 ],
                    [ 0.0,-0.5*math.sqrt(3), 0.5, 0.0 ],
                    [ 0.0, 0.5,-0.5*math.sqrt(3), 0.0 ],
                    [ 0.0,0.0,1.0,0.0 ],
                    [ 0.0,-0.5,-0.5*math.sqrt(3), 0.0 ],
                    [ 0.0, 0.5*math.sqrt(3), 0.5, 0.0 ],
                  ]
    elif self.lattice == 'tetragonal':
      symQuats =  [
                    [ 1.0,0.0,0.0,0.0 ],
                    [ 0.0,1.0,0.0,0.0 ],
                    [ 0.0,0.0,1.0,0.0 ],
                    [ 0.0,0.0,0.0,1.0 ],
                    [ 0.0, 0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0 ],
                    [ 0.0,-0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0 ],
                    [ 0.5*math.sqrt(2), 0.0, 0.0, 0.5*math.sqrt(2) ],
                    [-0.5*math.sqrt(2), 0.0, 0.0, 0.5*math.sqrt(2) ],
                  ]
    elif self.lattice == 'orthorhombic':
      symQuats =  [
                    [ 1.0,0.0,0.0,0.0 ],
                    [ 0.0,1.0,0.0,0.0 ],
                    [ 0.0,0.0,1.0,0.0 ],
                    [ 0.0,0.0,0.0,1.0 ],
                  ]
    else:
      symQuats =  [
                    [ 1.0,0.0,0.0,0.0 ],
                  ]

    return list(map(Quaternion,
               np.array(symQuats)[np.atleast_1d(np.array(who)) if who != [] else range(len(symQuats))]))
    
    
  def equivalentQuaternions(self,
                            quaternion,
                            who = []):
    """List of symmetrically equivalent quaternions based on own symmetry."""
    return [quaternion*q for q in self.symmetryQuats(who)]


  def inFZ(self,R):
    """Check whether given Rodrigues vector falls into fundamental zone of own symmetry."""
    if isinstance(R, Quaternion): R = R.asRodrigues()                                               # translate accidentially passed quaternion
    # fundamental zone in Rodrigues space is point symmetric around origin
    R = abs(R)                                                                                      
    if self.lattice == 'cubic':
      return     math.sqrt(2.0)-1.0 >= R[0] \
             and math.sqrt(2.0)-1.0 >= R[1] \
             and math.sqrt(2.0)-1.0 >= R[2] \
             and 1.0 >= R[0] + R[1] + R[2]
    elif self.lattice == 'hexagonal':
      return     1.0 >= R[0] and 1.0 >= R[1] and 1.0 >= R[2] \
             and 2.0 >= math.sqrt(3)*R[0] + R[1] \
             and 2.0 >= math.sqrt(3)*R[1] + R[0] \
             and 2.0 >= math.sqrt(3) + R[2]
    elif self.lattice == 'tetragonal':
      return     1.0 >= R[0] and 1.0 >= R[1] \
             and math.sqrt(2.0) >= R[0] + R[1] \
             and math.sqrt(2.0) >= R[2] + 1.0
    elif self.lattice == 'orthorhombic':
      return     1.0 >= R[0] and 1.0 >= R[1] and 1.0 >= R[2]
    else:
      return True


  def inDisorientationSST(self,R):
    """
    Check whether given Rodrigues vector (of misorientation) falls into standard stereographic triangle of own symmetry.
    
    Determination of disorientations follow the work of A. Heinz and P. Neumann:
    Representation of Orientation and Disorientation Data for Cubic, Hexagonal, Tetragonal and Orthorhombic Crystals
    Acta Cryst. (1991). A47, 780-789
    """
    if isinstance(R, Quaternion): R = R.asRodrigues()       # translate accidentially passed quaternion
    
    epsilon = 0.0
    if self.lattice == 'cubic':
      return R[0] >= R[1]+epsilon                and R[1] >= R[2]+epsilon    and R[2] >= epsilon
    
    elif self.lattice == 'hexagonal':
      return R[0] >= math.sqrt(3)*(R[1]-epsilon) and R[1] >= epsilon         and R[2] >= epsilon
    
    elif self.lattice == 'tetragonal':
      return R[0] >= R[1]-epsilon                and R[1] >= epsilon         and R[2] >= epsilon
    
    elif self.lattice == 'orthorhombic':
      return R[0] >= epsilon                     and R[1] >= epsilon         and R[2] >= epsilon
    
    else:
      return True


  def inSST(self,
            vector,
            proper = False,
            color = False):
    """
    Check whether given vector falls into standard stereographic triangle of own symmetry.

    proper considers only vectors with z >= 0, hence uses two neighboring SSTs.
    Return inverse pole figure color if requested.
    """
    #     basis = {'cubic' :        np.linalg.inv(np.array([[0.,0.,1.],                                 # direction of red
    #                                                       [1.,0.,1.]/np.sqrt(2.),                     # direction of green
    #                                                       [1.,1.,1.]/np.sqrt(3.)]).transpose()),      # direction of blue
    #              'hexagonal' :    np.linalg.inv(np.array([[0.,0.,1.],                                 # direction of red
    #                                                       [1.,0.,0.],                                 # direction of green
    #                                                       [np.sqrt(3.),1.,0.]/np.sqrt(4.)]).transpose()),      # direction of blue
    #              'tetragonal' :   np.linalg.inv(np.array([[0.,0.,1.],                                 # direction of red
    #                                                       [1.,0.,0.],                                 # direction of green
    #                                                       [1.,1.,0.]/np.sqrt(2.)]).transpose()),      # direction of blue
    #              'orthorhombic' : np.linalg.inv(np.array([[0.,0.,1.],                                 # direction of red
    #                                                       [1.,0.,0.],                                 # direction of green
    #                                                       [0.,1.,0.]]).transpose()),                  # direction of blue
    #             }

    if self.lattice == 'cubic':
      basis = {'improper':np.array([ [-1.            ,  0.            ,  1. ],
                                   [ np.sqrt(2.)   , -np.sqrt(2.)   ,  0. ],
                                   [ 0.            ,  np.sqrt(3.)   ,  0. ] ]),
             'proper':np.array([ [ 0.            , -1.            ,  1. ],
                                   [-np.sqrt(2.)   , np.sqrt(2.)    ,  0. ],
                                   [ np.sqrt(3.)   ,  0.            ,  0. ] ]),
              }
    elif self.lattice == 'hexagonal':
      basis = {'improper':np.array([ [ 0.            ,  0.            ,  1. ],
                                   [ 1.            , -np.sqrt(3.),     0. ],
                                   [ 0.            ,  2.            ,  0. ] ]),
             'proper':np.array([ [ 0.            ,  0.            ,  1. ],
                                   [-1.            ,  np.sqrt(3.)   ,  0. ],
                                   [ np.sqrt(3)    , -1.            ,  0. ] ]),
              }
    elif self.lattice == 'tetragonal':
      basis = {'improper':np.array([ [ 0.            ,  0.            ,  1. ],
                                   [ 1.            , -1.            ,  0. ],
                                   [ 0.            ,  np.sqrt(2.),     0. ] ]),
             'proper':np.array([ [ 0.            ,  0.            ,  1. ],
                                   [-1.            ,  1.            ,  0. ],
                                   [ np.sqrt(2.)   ,  0.            ,  0. ] ]),
              }
    elif self.lattice == 'orthorhombic':
      basis = {'improper':np.array([ [ 0., 0., 1.],
                                   [ 1., 0., 0.],
                                   [ 0., 1., 0.] ]),
             'proper':np.array([ [ 0., 0., 1.],
                                   [-1., 0., 0.],
                                   [ 0., 1., 0.] ]),
              }
    else:
      basis = {'improper':np.zeros((3,3),dtype=float),
             'proper':np.zeros((3,3),dtype=float),
              }

    if np.all(basis == 0.0):
      theComponents = -np.ones(3,'d')
      inSST = np.all(theComponents >= 0.0)
    else:
      v = np.array(vector,dtype = float)
      if proper:                                                                                    # check both improper ...
        theComponents = np.dot(basis['improper'],v)
        inSST = np.all(theComponents >= 0.0)
        if not inSST:                                                                               # ... and proper SST
          theComponents = np.dot(basis['proper'],v)
          inSST = np.all(theComponents >= 0.0)
      else:      
        v[2] = abs(v[2])                                                                            # z component projects identical 
        theComponents = np.dot(basis['improper'],v)                                                 # for positive and negative values
        inSST = np.all(theComponents >= 0.0)

    if color:                                                                                       # have to return color array
      if inSST:
        rgb = np.power(theComponents/np.linalg.norm(theComponents),0.5)                             # smoothen color ramps
        rgb = np.minimum(np.ones(3,'d'),rgb)                                                        # limit to maximum intensity
        rgb /= max(rgb)                                                                             # normalize to (HS)V = 1
      else:
        rgb = np.zeros(3,'d')
      return (inSST,rgb)
    else:
      return inSST

# code derived from http://pyeuclid.googlecode.com/svn/trunk/euclid.py
# suggested reading: http://web.mit.edu/2.998/www/QuaternionReport1.pdf



# ******************************************************************************************
class Orientation:

  __slots__ = ['quaternion','symmetry']

  def __init__(self,
               quaternion = Quaternion.fromIdentity(),
               Rodrigues  = None,
               angleAxis  = None,
               matrix     = None,
               Eulers     = None,
               random     = False,                                                                  # integer to have a fixed seed or True for real random
               symmetry   = None,
              ):
    if random:                                                                                      # produce random orientation
      if isinstance(random, bool ):
        self.quaternion = Quaternion.fromRandom()
      else:
        self.quaternion = Quaternion.fromRandom(randomSeed=random)
    elif isinstance(Eulers, np.ndarray) and Eulers.shape == (3,):                                   # based on given Euler angles
      self.quaternion = Quaternion.fromEulers(Eulers,'bunge')
    elif isinstance(matrix, np.ndarray) :                                                           # based on given rotation matrix
      self.quaternion = Quaternion.fromMatrix(matrix)
    elif isinstance(angleAxis, np.ndarray) and angleAxis.shape == (4,):                             # based on given angle and rotation axis
      self.quaternion = Quaternion.fromAngleAxis(angleAxis[0],angleAxis[1:4])
    elif isinstance(Rodrigues, np.ndarray) and Rodrigues.shape == (3,):                             # based on given Rodrigues vector
      self.quaternion = Quaternion.fromRodrigues(Rodrigues)
    elif isinstance(quaternion, Quaternion):                                                        # based on given quaternion
      self.quaternion = quaternion.homomorphed()
    elif isinstance(quaternion, np.ndarray) and quaternion.shape == (4,):                           # based on given quaternion
      self.quaternion = Quaternion(quaternion).homomorphed()

    self.symmetry = Symmetry(symmetry)

  def __copy__(self):
    """copy"""
    return self.__class__(quaternion=self.quaternion,symmetry=self.symmetry.lattice)

  copy = __copy__


  def __repr__(self):
    """value as all implemented representations"""
    return 'Symmetry: %s\n' % (self.symmetry) + \
           'Quaternion: %s\n' % (self.quaternion) + \
           'Matrix:\n%s\n' % ( '\n'.join(['\t'.join(list(map(str,self.asMatrix()[i,:]))) for i in range(3)]) ) + \
           'Bunge Eulers / deg: %s' % ('\t'.join(list(map(str,self.asEulers('bunge',degrees=True)))) )

  def asQuaternion(self):
    return self.quaternion.asList()
    
  # @property
  def asEulers(self,
               type = 'bunge',
               degrees = False,
               standardRange = False):
    return self.quaternion.asEulers(type, degrees, standardRange)
  eulers = property(asEulers)
  
  # @property
  def asRodrigues(self):
    return self.quaternion.asRodrigues()
  rodrigues = property(asRodrigues)
  
  # @property
  def asAngleAxis(self, degrees = False):
    return self.quaternion.asAngleAxis(degrees)
  angleAxis = property(asAngleAxis)

  # @property
  def asMatrix(self):
    return self.quaternion.asMatrix()
  matrix = property(asMatrix)
  
  # @property
  def inFZ(self):
    return self.symmetry.inFZ(self.quaternion.asRodrigues())
  infz = property(inFZ)

  def equivalentQuaternions(self,
                            who = []):
    return self.symmetry.equivalentQuaternions(self.quaternion,who)

  def equivalentOrientations(self,
                             who = []):
    return list(map(lambda q: Orientation(quaternion = q, symmetry = self.symmetry.lattice),
               self.equivalentQuaternions(who)))

  def reduced(self):
    """Transform orientation to fall into fundamental zone according to symmetry"""
    for me in self.symmetry.equivalentQuaternions(self.quaternion):
      if self.symmetry.inFZ(me.asRodrigues()): break

    return Orientation(quaternion=me,symmetry=self.symmetry.lattice)


  def disorientation(self,
                     other,
                     SST = True):
    """
    Disorientation between myself and given other orientation.

    Rotation axis falls into SST if SST == True.
    (Currently requires same symmetry for both orientations.
     Look into A. Heinz and P. Neumann 1991 for cases with differing sym.)
    """
    if self.symmetry != other.symmetry: raise TypeError('disorientation between different symmetry classes not supported yet.')

    misQ = self.quaternion.conjugated()*other.quaternion
    mySymQs    =  self.symmetry.symmetryQuats() if SST else self.symmetry.symmetryQuats()[:1]       # take all or only first sym operation
    otherSymQs = other.symmetry.symmetryQuats()
    
    for i,sA in enumerate(mySymQs):
      for j,sB in enumerate(otherSymQs):
        theQ = sA.conjugated()*misQ*sB
        for k in range(2):
          theQ.conjugate()
          breaker = self.symmetry.inFZ(theQ) \
                    and (not SST or other.symmetry.inDisorientationSST(theQ))
          if breaker: break
        if breaker: break
      if breaker: break

    # disorientation, own sym, other sym, self-->other: True, self<--other: False
    return (Orientation(quaternion = theQ, symmetry = self.symmetry.lattice),
            i,j,k == 1)                                                                             


  def inversePole(self,
                  axis,
                  proper = False,
                  SST = True):
    """axis rotated according to orientation (using crystal symmetry to ensure location falls into SST)"""
    if SST:                                                                                         # pole requested to be within SST
      for i,q in enumerate(self.symmetry.equivalentQuaternions(self.quaternion)):                   # test all symmetric equivalent quaternions
        pole = q.conjugated()*axis                                                                  # align crystal direction to axis
        if self.symmetry.inSST(pole,proper): break                                                  # found SST version
    else:
      pole = self.quaternion.conjugated()*axis                                                      # align crystal direction to axis
    return (pole,i if SST else 0)

  def IPFcolor(self,axis):
    """TSL color of inverse pole figure for given axis"""
    color = np.zeros(3,'d')

    for q in self.symmetry.equivalentQuaternions(self.quaternion):
      pole = q.conjugated()*axis                                                                    # align crystal direction to axis
      inSST,color = self.symmetry.inSST(pole,color=True)
      if inSST: break

    return color

  @classmethod
  def average(cls,
              orientations,
              multiplicity = []):
    """
    average orientation

    ref: F. Landis Markley, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
         Averaging Quaternions,
         Journal of Guidance, Control, and Dynamics, Vol. 30, No. 4 (2007), pp. 1193-1197.
         doi: 10.2514/1.28949
    usage:
         a = Orientation(Eulers=np.radians([10, 10, 0]), symmetry='hexagonal')
         b = Orientation(Eulers=np.radians([20, 0, 0]),  symmetry='hexagonal')
         avg = Orientation.average([a,b])
    """
    if not all(isinstance(item, Orientation) for item in orientations):
      raise TypeError("Only instances of Orientation can be averaged.")

    N = len(orientations)
    if multiplicity == [] or not multiplicity:
      multiplicity = np.ones(N,dtype='i')

    reference = orientations[0]                                                                     # take first as reference
    for i,(o,n) in enumerate(zip(orientations,multiplicity)):
      closest = o.equivalentOrientations(reference.disorientation(o,SST = False)[2])[0]             # select sym orientation with lowest misorientation
      M = closest.quaternion.asM() * n if i == 0 else M + closest.quaternion.asM() * n              # noqa add (multiples) of this orientation to average noqa
    eig, vec = np.linalg.eig(M/N)

    return Orientation(quaternion = Quaternion(quatArray = np.real(vec.T[eig.argmax()])),
                       symmetry = reference.symmetry.lattice)